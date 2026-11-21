package com.precipitation.controller;

import com.precipitation.model.ApiResponse;
import com.precipitation.model.PredictionRequest;
import com.precipitation.model.PredictionResult;
import com.precipitation.service.PredictionService;
import jakarta.validation.Valid;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

/**
 * 降水预测控制器
 */
@Slf4j
@RestController
@RequestMapping("/prediction")
public class PredictionController {
    
    private final PredictionService predictionService;
    
    public PredictionController(PredictionService predictionService) {
        this.predictionService = predictionService;
    }
    
    /**
     * 执行预测
     */
    @PostMapping(value = "/predict", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<ApiResponse<PredictionResult>> predict(
            @RequestParam("files") MultipartFile[] files,
            @Valid @ModelAttribute PredictionRequest request) {
        
        log.info("收到预测请求，文件数量: {}, 参数: {}", files.length, request);
        
        try {
            PredictionResult result = predictionService.predict(files, request);
            return ResponseEntity.ok(ApiResponse.success("预测成功", result));
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().body(ApiResponse.error(400, e.getMessage()));
        } catch (Exception e) {
            log.error("预测失败", e);
            return ResponseEntity.status(502).body(ApiResponse.error(502, "推理服务未能完成请求"));
        }
    }
    
    /**
     * 查询预测结果
     */
    @GetMapping("/result/{taskId}")
    public ResponseEntity<ApiResponse<PredictionResult>> getResult(@PathVariable String taskId) {
        log.info("查询任务结果: {}", taskId);
        
        try {
            PredictionResult result = predictionService.getResult(taskId);
            return ResponseEntity.ok(ApiResponse.success(result));
        } catch (java.io.FileNotFoundException | IllegalArgumentException e) {
            return ResponseEntity.status(404).body(ApiResponse.error(404, "任务不存在"));
        } catch (Exception e) {
            log.error("查询失败", e);
            return ResponseEntity.internalServerError().body(ApiResponse.error("读取任务失败"));
        }
    }
    
    /**
     * 健康检查
     */
    @GetMapping("/health")
    public ApiResponse<String> health() {
        return ApiResponse.success("服务运行正常");
    }
}

