package com.precipitation.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.precipitation.config.FileStorageConfig;
import com.precipitation.model.PredictionRequest;
import com.precipitation.model.PredictionResult;
import com.precipitation.util.HttpClientUtil;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.*;

/**
 * 预测业务服务类
 */
@Slf4j
@Service
public class PredictionService {
    
    @Value("${ai.service.url}")
    private String aiServiceUrl;
    
    @Value("${ai.service.predict-endpoint}")
    private String predictEndpoint;
    
    @Value("${ai.service.timeout}")
    private Integer timeout;
    
    private final FileStorageConfig fileStorageConfig;
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    public PredictionService(FileStorageConfig fileStorageConfig) {
        this.fileStorageConfig = fileStorageConfig;
    }
    
    /**
     * 执行预测
     */
    public PredictionResult predict(MultipartFile[] files, PredictionRequest request) throws Exception {
        log.info("开始预测任务，文件数量: {}", files.length);
        
        // 1. 验证文件
        validateFiles(files);
        
        // 2. 生成任务ID
        String taskId = UUID.randomUUID().toString();
        
        // 3. 保存上传的文件
        File uploadDir = new File(fileStorageConfig.getUploadDir(), taskId);
        uploadDir.mkdirs();
        
        List<File> savedFiles = new ArrayList<>();
        for (int i = 0; i < files.length; i++) {
            MultipartFile file = files[i];
            String filename = String.format("input_%03d.npy", i);
            Path targetPath = Paths.get(uploadDir.getAbsolutePath(), filename);
            Files.copy(file.getInputStream(), targetPath, StandardCopyOption.REPLACE_EXISTING);
            savedFiles.add(targetPath.toFile());
            log.info("保存文件: {}", targetPath);
        }
        
        // 4. 调用AI服务进行预测
        long startTime = System.currentTimeMillis();
        PredictionResult result = callAIService(taskId, savedFiles, request);
        long endTime = System.currentTimeMillis();
        
        result.setProcessingTime((endTime - startTime) / 1000.0);
        result.setTaskId(taskId);
        
        log.info("预测完成，任务ID: {}, 耗时: {}秒", taskId, result.getProcessingTime());
        
        return result;
    }
    
    /**
     * 验证上传的文件
     */
    private void validateFiles(MultipartFile[] files) {
        if (files == null || files.length == 0) {
            throw new IllegalArgumentException("请上传至少一个NPY文件");
        }
        
        for (MultipartFile file : files) {
            if (file.isEmpty()) {
                throw new IllegalArgumentException("文件不能为空");
            }
            
            String filename = file.getOriginalFilename();
            if (filename == null || !filename.toLowerCase().endsWith(".npy")) {
                throw new IllegalArgumentException("只支持NPY格式文件");
            }
            
            if (file.getSize() > fileStorageConfig.getMaxFileSize()) {
                throw new IllegalArgumentException("文件大小不能超过100MB");
            }
        }
    }
    
    /**
     * 调用AI服务
     */
    private PredictionResult callAIService(String taskId, List<File> files, PredictionRequest request) throws Exception {
        String url = aiServiceUrl + predictEndpoint;
        
        // 准备参数
        Map<String, Object> params = new HashMap<>();
        params.put("task_id", taskId);
        params.put("input_length", request.getInputLength());
        params.put("predict_length", request.getPredictLength());
        params.put("generate_comparison", request.getGenerateComparison());
        
        // 发送第一个文件（或压缩包）
        File firstFile = files.get(0);
        
        try {
            String responseJson = HttpClientUtil.postMultipart(url, params, firstFile, timeout / 1000);
            log.info("AI服务响应: {}", responseJson);
            
            // 解析响应
            return parseAIResponse(responseJson, taskId);
        } catch (Exception e) {
            log.error("调用AI服务失败", e);
            return PredictionResult.builder()
                    .taskId(taskId)
                    .status("failed")
                    .message("预测失败: " + e.getMessage())
                    .build();
        }
    }
    
    /**
     * 解析AI服务响应
     */
    private PredictionResult parseAIResponse(String jsonResponse, String taskId) throws IOException {
        JsonNode root = objectMapper.readTree(jsonResponse);
        
        PredictionResult.PredictionResultBuilder builder = PredictionResult.builder()
                .taskId(taskId)
                .status(root.path("status").asText("success"))
                .message(root.path("message").asText("预测完成"));
        
        // 解析图像URL
        List<String> inputImages = new ArrayList<>();
        JsonNode inputNode = root.path("input_images");
        if (inputNode.isArray()) {
            inputNode.forEach(node -> inputImages.add(node.asText()));
        }
        builder.inputImages(inputImages);
        
        List<String> predictionImages = new ArrayList<>();
        JsonNode predNode = root.path("prediction_images");
        if (predNode.isArray()) {
            predNode.forEach(node -> predictionImages.add(node.asText()));
        }
        builder.predictionImages(predictionImages);
        
        // 对比图和下载链接
        if (root.has("comparison_image")) {
            builder.comparisonImage(root.path("comparison_image").asText());
        }
        
        if (root.has("download_url")) {
            builder.downloadUrl(root.path("download_url").asText());
        }
        
        // 指标
        if (root.has("metrics")) {
            JsonNode metrics = root.path("metrics");
            if (metrics.has("mse")) {
                builder.mse(metrics.path("mse").asDouble());
            }
            if (metrics.has("ssim")) {
                builder.ssim(metrics.path("ssim").asDouble());
            }
        }
        
        return builder.build();
    }
    
    /**
     * 获取任务结果
     */
    public PredictionResult getResult(String taskId) {
        // 检查结果目录是否存在
        File resultDir = new File(fileStorageConfig.getResultDir(), taskId);
        
        if (!resultDir.exists()) {
            return PredictionResult.builder()
                    .taskId(taskId)
                    .status("not_found")
                    .message("任务不存在")
                    .build();
        }
        
        // 构建结果
        return PredictionResult.builder()
                .taskId(taskId)
                .status("success")
                .message("查询成功")
                .build();
    }
}

