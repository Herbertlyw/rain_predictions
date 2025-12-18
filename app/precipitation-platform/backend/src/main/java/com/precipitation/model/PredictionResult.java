package com.precipitation.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * 预测结果
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PredictionResult {
    
    /**
     * 任务ID
     */
    private String taskId;
    
    /**
     * 预测状态：success/failed/processing
     */
    private String status;
    
    /**
     * 消息
     */
    private String message;
    
    /**
     * 输入图像URL列表
     */
    private List<String> inputImages;
    
    /**
     * 预测图像URL列表
     */
    private List<String> predictionImages;
    
    /**
     * 对比图URL
     */
    private String comparisonImage;
    
    /**
     * 下载链接
     */
    private String downloadUrl;
    
    /**
     * 处理时间（秒）
     */
    private Double processingTime;
    
    /**
     * MSE指标
     */
    private Double mse;
    
    /**
     * SSIM指标
     */
    private Double ssim;
}

