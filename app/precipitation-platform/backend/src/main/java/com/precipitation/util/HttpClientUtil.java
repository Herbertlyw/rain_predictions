package com.precipitation.util;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.client5.http.config.RequestConfig;
import org.apache.hc.client5.http.entity.mime.MultipartEntityBuilder;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.CloseableHttpResponse;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.http.ContentType;
import org.apache.hc.core5.http.HttpEntity;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.hc.core5.util.Timeout;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * HTTP客户端工具类
 */
@Component
public class HttpClientUtil {
    
    private static final ObjectMapper objectMapper = new ObjectMapper();
    
    /**
     * 发送multipart/form-data POST请求
     */
    public static String postMultipart(String url, Map<String, Object> params, File file, int timeoutSeconds) throws IOException {
        // 配置超时
        RequestConfig requestConfig = RequestConfig.custom()
                .setConnectTimeout(Timeout.of(10, TimeUnit.SECONDS))
                .setResponseTimeout(Timeout.of(timeoutSeconds, TimeUnit.SECONDS))
                .build();
        
        try (CloseableHttpClient httpClient = HttpClients.custom()
                .setDefaultRequestConfig(requestConfig)
                .build()) {
            
            HttpPost httpPost = new HttpPost(url);
            
            // 构建multipart实体
            MultipartEntityBuilder builder = MultipartEntityBuilder.create();
            
            // 添加文件
            if (file != null && file.exists()) {
                builder.addBinaryBody("file", file, ContentType.APPLICATION_OCTET_STREAM, file.getName());
            }
            
            // 添加其他参数
            if (params != null) {
                params.forEach((key, value) -> {
                    if (value != null) {
                        builder.addTextBody(key, value.toString(), ContentType.TEXT_PLAIN);
                    }
                });
            }
            
            HttpEntity entity = builder.build();
            httpPost.setEntity(entity);
            
            // 执行请求
            try (CloseableHttpResponse response = httpClient.execute(httpPost)) {
                HttpEntity responseEntity = response.getEntity();
                return EntityUtils.toString(responseEntity, "UTF-8");
            }
        }
    }
    
    /**
     * 解析JSON响应
     */
    public static <T> T parseResponse(String jsonStr, Class<T> clazz) throws IOException {
        return objectMapper.readValue(jsonStr, clazz);
    }
}

