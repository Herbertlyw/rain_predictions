package com.precipitation.config;

import jakarta.annotation.PostConstruct;
import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import java.io.File;

/**
 * 文件存储配置
 */
@Data
@Configuration
@ConfigurationProperties(prefix = "file.storage")
public class FileStorageConfig {
    
    private String uploadDir;
    private String resultDir;
    private Long maxFileSize;
    
    @PostConstruct
    public void init() {
        // 创建目录
        new File(uploadDir).mkdirs();
        new File(resultDir).mkdirs();
    }
}

