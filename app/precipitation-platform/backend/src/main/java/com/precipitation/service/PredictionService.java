package com.precipitation.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.precipitation.config.FileStorageConfig;
import com.precipitation.model.PredictionRequest;
import com.precipitation.model.PredictionResult;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;
import java.util.stream.StreamSupport;

@Service
public class PredictionService {
    private final RestTemplate http;
    private final String aiUrl;
    private final String predictEndpoint;
    private final Path results;
    private final ObjectMapper json;

    public PredictionService(@Value("${ai.service.url}") String aiUrl,
                             @Value("${ai.service.predict-endpoint}") String predictEndpoint,
                             @Value("${ai.service.timeout}") int timeout,
                             FileStorageConfig storage, ObjectMapper json) {
        this.aiUrl = aiUrl;
        this.predictEndpoint = predictEndpoint;
        this.results = Path.of(storage.getResultDir());
        this.json = json;
        var factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(10_000);
        factory.setReadTimeout(timeout);
        this.http = new RestTemplate(factory);
    }

    public PredictionResult predict(MultipartFile[] files, PredictionRequest request) throws IOException {
        if (files.length != request.getInputLength()) throw new IllegalArgumentException("文件数量与输入帧数不一致");
        UUID id = UUID.randomUUID();
        var parts = new LinkedMultiValueMap<String, Object>();
        for (MultipartFile file : files) {
            if (file.isEmpty() || file.getOriginalFilename() == null ||
                    !file.getOriginalFilename().toLowerCase(java.util.Locale.ROOT).endsWith(".npy")) {
                throw new IllegalArgumentException("请选择非空 NPY 文件");
            }
            parts.add("file", file.getResource());
        }
        parts.add("task_id", id.toString());
        parts.add("input_length", request.getInputLength().toString());
        parts.add("predict_length", request.getPredictLength().toString());
        parts.add("generate_comparison", request.getGenerateComparison().toString());
        var headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        long start = System.nanoTime();
        JsonNode reply;
        try {
            reply = http.postForObject(aiUrl + predictEndpoint, new HttpEntity<>(parts, headers), JsonNode.class);
        } catch (HttpClientErrorException.BadRequest e) {
            throw new IllegalArgumentException(json.readTree(e.getResponseBodyAsString()).path("message").asText("输入无效"));
        }
        if (reply == null || !"success".equals(reply.path("status").asText())) throw new IOException("模型服务预测失败");
        PredictionResult result = PredictionResult.builder().taskId(id.toString()).status("success").message("预测完成")
                .processingTime((System.nanoTime() - start) / 1e9)
                .inputImages(links(reply.path("input_images"))).predictionImages(links(reply.path("prediction_images")))
                .comparisonImage(reply.hasNonNull("comparison_image") ? aiUrl + reply.get("comparison_image").asText() : null)
                .downloadUrl(aiUrl + reply.path("download_url").asText()).build();
        Path draft = Files.createTempFile(results, id.toString(), ".tmp");
        try {
            json.writeValue(draft.toFile(), result);
            Files.move(draft, results.resolve(id + ".json"), java.nio.file.StandardCopyOption.ATOMIC_MOVE);
        } finally {
            Files.deleteIfExists(draft);
        }
        return result;
    }

    private List<String> links(JsonNode array) {
        return StreamSupport.stream(array.spliterator(), false).map(node -> aiUrl + node.asText()).toList();
    }

    public PredictionResult getResult(String taskId) throws IOException {
        Path file = results.resolve(UUID.fromString(taskId) + ".json");
        if (!Files.isRegularFile(file)) throw new java.io.FileNotFoundException("任务不存在");
        return json.readValue(file.toFile(), PredictionResult.class);
    }
}
