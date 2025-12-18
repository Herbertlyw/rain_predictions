package com.precipitation;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * 降水预测平台主应用程序
 * 
 * @author 项目组
 * @version 1.0
 */
@SpringBootApplication
public class PrecipitationApplication {

    public static void main(String[] args) {
        SpringApplication.run(PrecipitationApplication.class, args);
        System.out.println("===========================================");
        System.out.println("降水预测平台启动成功！");
        System.out.println("访问地址: http://localhost:8080/api");
        System.out.println("===========================================");
    }
}

