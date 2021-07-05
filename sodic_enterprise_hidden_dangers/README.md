# 企业隐患排查文本挖掘

## 1、比赛说明

[基于文本挖掘的企业隐患排查质量分析模型](https://www.sodic.com.cn/competitions/900010)

![image-20210628232157298](D:\develop\github\Competition_Road\sodic_enterprise_hidden_dangers\img\20.jpg)

### 1.1 赛题背景

企业自主填报安全生产隐患，对于将风险消除在事故萌芽阶段具有重要意义。企业在填报隐患时，往往存在不认真填报的情况，“虚报、假报”隐患内容，增大了企业监管的难度。采用大数据手段分析隐患内容，找出不切实履行主体责任的企业，向监管部门进行推送，实现精准执法，能够提高监管手段的有效性，增强企业安全责任意识。

### 1.2 赛题任务

本赛题提供企业填报隐患数据，参赛选手需通过智能化手段识别其中是否存在“虚报、假报”的情况。

### 1.3 赛题数据

**训练集：**



![image-20210628213403267](D:\develop\github\Competition_Road\sodic_enterprise_hidden_dangers\img\9.jpg)

**测试集：**

![image-20210628213509197](D:\develop\github\Competition_Road\sodic_enterprise_hidden_dangers\img\10.jpg)

**数据说明：**

训练集数据包含“【id、level_1（一级标准）、level_2（二级标准）、level_3（三级标准）、level_4（四级标准）、content（隐患内容）和label（标签）】”共7个字段。

其中“id”为主键，无业务意义；“一级标准、二级标准、三级标准、四级标准”为《深圳市安全隐患自查和巡查基本指引（2016年修订版）》规定的排查指引，一级标准对应不同隐患类型，二至四级标准是对一级标准的细化，企业自主上报隐患时，根据不同类型隐患的四级标准开展隐患自查工作；“隐患内容”为企业上报的具体隐患；“标签”标识的是该条隐患的合格性，“1”表示隐患填报不合格，“0”表示隐患填报合格。

**预测结果文件：**

![image-20210628213730940](D:\develop\github\Competition_Road\sodic_enterprise_hidden_dangers\img\11.jpg)

### 1.4 评测标准

![image-20210628213824015](D:\develop\github\Competition_Road\sodic_enterprise_hidden_dangers\img\12.jpg)

## 2、比赛总结

| 编号 | 内容            | 博客                                                         |
| ---- | --------------- | ------------------------------------------------------------ |
| 1    | 数据处理        | [企业隐患排查文本挖掘比赛（一）：数据篇](https://blog.csdn.net/olizxq/article/details/118345296) |
| 2    | BERT子模型      | [企业隐患排查文本挖掘比赛（二）：算法篇（从词向量到BERT）](https://blog.csdn.net/olizxq/article/details/118420358) |
| 3    | LGB文本分类模型 | [企业隐患排查文本挖掘比赛（三）：LGB文本分类(调参+阈值搜索)](https://blog.csdn.net/olizxq/article/details/118463493) |

