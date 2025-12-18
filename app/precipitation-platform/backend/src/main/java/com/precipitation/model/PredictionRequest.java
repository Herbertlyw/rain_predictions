package com.precipitation.model;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * 预测请求参数
 */
@Data
public class PredictionRequest {
    
    /**
     * 输入帧数量
     */
    @NotNull(message = "输入帧数量不能为空")
    @Min(value = 1, message = "输入帧数量至少为1")
    @Max(value = 20, message = "输入帧数量最多为20")
    private Integer inputLength = 5;
    
    /**
     * 预测帧数量
     */
    @NotNull(message = "预测帧数量不能为空")
    @Min(value = 1, message = "预测帧数量至少为1")
    @Max(value = 20, message = "预测帧数量最多为20")
    private Integer predictLength = 5;
    
    /**
     * 是否生成对比图
     */
    private Boolean generateComparison = true;
}

