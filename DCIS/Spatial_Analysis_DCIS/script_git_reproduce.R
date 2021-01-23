library(ggpubr)
library(ggplot2)
library(gridExtra)

#Figure 4

t=read.csv("Dave_validData_Fig4_data.csv")

p1 <- ggscatter(t,x="B.sz.IHC",y="A.sz.HE", add="reg.line",conf.int=TRUE,cor.coef = TRUE,cor.method = "spearman",xlab="Biological DCIS estimation",ylab="Automated DCIS estimation")
#dev.off()

t=read.csv("Dave_validData_annotation_area.csv")
pdf(file = "figures/Fig4d.pdf", width = 18 , height = 8)
p2 <- ggscatter(t,x="IHC",y="HE", add="reg.line",conf.int=TRUE,cor.coef = TRUE, cor.method = "pearson",xlab="Biological DCIS estimation",ylab="Automated DCIS estimation")
p3<-grid.arrange(p1,p2, nrow=1)
dev.off()

#Figure 6

MS_data=read.csv('Dave_validData_Fig6_data.csv',stringsAsFactors = F)
MS_data_PD=MS_data[MS_data$Group == "Pure DCIS",]
MS_data_AD=MS_data[MS_data$Group == "Adjacent DCIS",]
pdf(file="figures/MS_Lym_Pure_adjacent_Figure6.pdf",width = 12 , height = 8)

p1 <- ggboxplot(MS_data,x="Group",y="DCIS.MS", color="Group", palette = c("#0000FF", "#FC4E07"), line.color="gray", line.size=1.0, add="jitter")+scale_y_continuous(limits=c(0,1))+
  stat_compare_means(method="wilcox")+theme(text = element_text(size=18))

p2 <- ggboxplot(MS_data,x="Group",y="LP", color="Group", palette = c("#0000FF", "#FC4E07"), line.color="gray", line.size=1.0, add="jitter")+scale_y_continuous(limits=c(0,1))+
  stat_compare_means(method="wilcox")+theme(text = element_text(size=18))



p3<-grid.arrange(p1,p2, nrow=1)

dev.off()
