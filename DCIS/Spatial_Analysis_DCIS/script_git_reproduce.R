library(ggpubr)
library(ggplot2)
library(gridExtra)

# Figure 4

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


tab=read.csv("CD4_CD8_FOXP3_data.csv", as.is=T)
x=log(tab$number.of.CD4.cells-tab$number.of.CD4.cells.in.DCIS)
y=log(tab$number.of.CD4.cells.in.DCIS)


Boxplot = function(x, y, main=NULL, labels=NULL,...){
  x=log(x)
  y=log(y)
  ks <- t.test(x,y, paired=T, ...)
  p <- signif((ks)$p.value,2)
  boxplot(x, y, main=paste(main,'p=', p), xlab="", ylab="Log cell number", varwidth=TRUE, border=c("steelblue2", "yellow2"),...)
  points(x=c(rep(1,7),rep(2,7)), y=c(x,y), col=c(rep("steelblue2",7), rep("yellow2", 7)), pch=19)
  for(i in 1:7)
    lines(x=1:2, y=cbind(x,y)[i,], col="grey")
  axis(side=1, at=1:2,labels=labels)
  return (p)
}


Boxplot_m = function(x, y, main=NULL, labels=NULL,...){
  x=x
  y=y
  ks <- t.test(x,y, paired=T, ...)
  p <- signif((ks)$p.value,2)
  boxplot(x, y, main=paste(main,'p=', p), xlab="", ylab="Morisita",  varwidth=TRUE, border=c("steelblue2", "yellow2"),...)
  points(x=c(rep(1,7),rep(2,7)), y=c(x,y), col=c(rep("steelblue2",7), rep("yellow2", 7)), pch=19)
  for(i in 1:7)
    lines(x=1:2, y=cbind(x,y)[i,], col="grey")
  axis(side=1, at=1:2,labels=labels)
  return (p)
}

pdf(file="figures/boxplot_inv_dcis_number_wil_inv_dcis.pdf", width=8, height=3)
#png(paste0("1//boxplot_inv_dcis_number_wil_inv_dcis.png"),width=18, height=8, units = "cm", res=100)
par(mfrow=c(1,3))
p1n=Boxplot((tab$number.of.CD4.cells-tab$number.of.CD4.cells.in.DCIS),(tab$number.of.CD4.cells.in.DCIS), main='CD4', labels=c("Invasive", "DCIS"))
p2n=Boxplot((tab$number.of.CD8.cells-tab$number.of.CD8.cells.in.DCIS), tab$number.of.CD8.cells.in.DCIS, main='CD8', labels=c("Invasive", "DCIS"))
p3n=Boxplot(tab$number.of.foxp3.cells-tab$number.of.foxp3.cells.in.DCIS, tab$number.of.foxp3.cells.in.DCIS, main='Foxp3', labels=c("Invasive", "DCIS"))
p=c(p1n,p2n,p3n)
p_adjust_cell_number_BH<-p.adjust(p, method = "BH")
p_adjust_cell_number_BH
dev.off()


pdf(file="figures/boxplot_inv_dcis_ratio_wil.pdf", width=8, height=3)
par(mfrow=c(1,3))
p1n=Boxplot((tab$number.of.CD8.cells-tab$number.of.CD8.cells.in.DCIS)/(tab$number.of.foxp3.cells-tab$number.of.foxp3.cells.in.DCIS), tab$number.of.CD8.cells.in.DCIS/tab$number.of.foxp3.cells.in.DCIS, main='CD8/FOXP3', labels=c("Invasive", "DCIS"))
p2n=Boxplot((tab$number.of.CD4.cells-tab$number.of.CD4.cells.in.DCIS)/(tab$number.of.foxp3.cells-tab$number.of.foxp3.cells.in.DCIS), tab$number.of.CD4.cells.in.DCIS/tab$number.of.foxp3.cells.in.DCIS, main='CD4/FOXP3', labels=c("Invasive", "DCIS"))
p3n=Boxplot((tab$number.of.CD8.cells-tab$number.of.CD8.cells.in.DCIS)/(tab$number.of.CD4.cells-tab$number.of.CD4.cells.in.DCIS), tab$number.of.CD8.cells.in.DCIS/tab$number.of.CD4.cells.in.DCIS, main='CD8/CD4', labels=c("Invasive", "DCIS"))
#Boxplot((tab$number.of.CD8.cells-tab$number.of.CD4.cells.in.DCIS)/(tab$number.of.CD8.cells-tab$number.of.CD8.cells.in.DCIS), tab$number.of.CD4.cells.in.DCIS/tab$number.of.CD8.cells.in.DCIS, main='CD4/CD8', labels=c("Invasive", "DCIS"))
p=c(p1n,p2n,p3n)
p_adjust_cell_type_ratio_BH<-p.adjust(p, method = "BH")
p_adjust_cell_type_ratio_BH
dev.off()


pdf(file="figures/boxplot_inv_dcis_morisita_phenmap.pdf", width=8, height=3)
#png(paste0("1//boxplot_inv_dcis_morisita_wil.png"),width=18, height=8, units = "cm", res=100)
par(mfrow=c(1,3))
p1=Boxplot_m((tab$morisita.cd4),(tab$morisita.cd8),main='Morisita',labels=c('CD4','CD8'))
p2=Boxplot_m((tab$morisita.cd4),(tab$morisita.foxp3),main='Morisita',labels=c('CD4','Foxp3'))
p3=Boxplot_m((tab$morisita.foxp3),(tab$morisita.cd8), main='Morisita',labels=c('Foxp3','CD8'))

p=c(p1,p2,p3)
p_adjust_dcis<-p.adjust(p, method = "BH")
p_adjust_dcis
#resulting p from BH 0.000165 0.730000 0.000036
#resulting p from hochberg  2.2e-04 7.3e-01 3.6e-05

dev.off()

pdf(file="figures/boxplot_morisita_dcis_inv.pdf", width=8, height=3)
#png(paste0("1//boxplot_inv_dcis_morisita_dcis_inv_v1.png"),width=18, height=8, units = "cm", res=100)
par(mfrow=c(1,3))
p1=Boxplot_m((tab$morisita.cd4.inv),(tab$morisita.cd4), main='Morisita CD4',labels=c("Invasive", "DCIS"))
p2=Boxplot_m((tab$morisita.cd8.inv),(tab$morisita.cd8), main='Morisita CD8',labels=c("Invasive", "DCIS"))
p3=Boxplot_m((tab$morisita.foxp3.inv),(tab$morisita.foxp3), main='Morisita FoxP3',labels=c("Invasive", "DCIS"))

p=c(p1,p2,p3)
p_adjust_inv_dcis<-p.adjust(p, method = "BH")
p_adjust_inv_dcis
# resulting p from BH and hochberg are same  0.41000 0.00027 0.03150
# resulting p from hochberg is  0.41000 0.00027 0.04200

dev.off()



tab1=tab[,c(3:4)]-tab[,10:11]
barplot(as.matrix(t(cbind(tab1, tab[,c(8:10)], tab[,7]-rowSums(tab[,2:4])))))
barplot(tab[,10]/tab[,8])

cols=c("darkred", "steelblue", "grey")
pdf(file="figures/barplot.pdf", width=4, height=9)
#png(paste0("1//barplot.png"),width=12, height=18, units = "cm", res=100)
par(mfrow=c(3,1))
x=tab[,3:4]/tab[,7]
x=cbind(x,1-rowSums(x))
barplot(t(x), beside=F, col= cols, ylab='Proportion of lymphocyte',  xlab="WSI")
legend('topright', inset=c(-0.01,0), legend=c('CD8', 'Foxp3', 'others'), col=cols, pch=15)

x=tab[,10:11]/tab[,7]
x=cbind(x,1-rowSums(x))
barplot(t(x), beside=F, col= cols, ylab='Proportion of lymphocyte',  xlab="DCIS")
legend('topright', inset=c(-0.01,0), legend=c('CD8', 'Foxp3', 'others'), col= cols, pch=15)

x=tab[,3:4]-tab[,10:11]
x=x/tab[,7]
x=cbind(x,1-rowSums(x))
barplot(t(x), beside=F, col= cols, ylab='Proportion of lymphocyte',  xlab="Invasive")
legend('topright', inset=c(-0.01,0), legend=c('CD8', 'Foxp3', 'others'), col= cols, pch=15)
dev.off()

library("ggpubr")
library(ggpubr,magittr)
library("ggplot2")

pdf(file="figures/correlation_CD4_CD8_FoxP3_HE_WSI.pdf",width=8,height=8)
cor_HE_CD4 <- tab[, c(2:4 ,7)]
p1_inv <- ggscatter(cor_HE_CD4,x="number.of.lym.in.H.E",y="number.of.CD4.cells",add="reg.line",conf.int=TRUE,cor.coef = TRUE,cor.method = "spearman", xlab="HE", ylab="CD4")
#dev.off()

#pdf(file="1/correlation_CD8_HE.pdf",width=12,height=8)
p2_inv <- ggscatter(cor_HE_CD4,x="number.of.lym.in.H.E",y="number.of.CD8.cells",add="reg.line",conf.int=TRUE,cor.coef = TRUE,cor.method = "spearman", xlab="HE", ylab="CD8")
#dev.off()

#pdf(file="1/correlation_FoxP3_HE.pdf",width=12,height=8)
p3_inv <- ggscatter(cor_HE_CD4,x="number.of.lym.in.H.E",y="number.of.foxp3.cells",add="reg.line",conf.int=TRUE,cor.coef = TRUE,cor.method = "spearman", xlab="HE", ylab="FoxP3")
ggarrange(p1_inv,p2_inv,p3_inv,nrow=1)
dev.off()
#dev.off()


pdf(file="figures/correlation_CD4_CD8_FoxP3_HE_DCIS.pdf",width=10,height=6)
cor_HE_CD4_DCIS <- tab[, c(2:4 ,8)]
p1 <- ggscatter(cor_HE_CD4_DCIS,x="number.of.lym.in.H.E.DCIS",y="number.of.CD4.cells",add="reg.line",conf.int=TRUE,cor.coef = TRUE,cor.method = "spearman", xlab="HE", ylab="CD4")
#dev.off()

#pdf(file="1/correlation_CD8_HE_DCIS.pdf",width=12,height=8)
#cor_HE_CD4_DCIS <- tab[, c(2:4 ,8)]
p2 <- ggscatter(cor_HE_CD4_DCIS,x="number.of.lym.in.H.E.DCIS",y="number.of.CD8.cells",add="reg.line",conf.int=TRUE,cor.coef = TRUE,cor.method = "spearman", xlab="HE", ylab="CD8")
#dev.off()

#pdf(file="1/correlation_FoxP3_HE_DCIS.pdf",width=12,height=8)
#cor_HE_CD4_DCIS <- tab[, c(2:4 ,8)]
p3 <- ggscatter(cor_HE_CD4_DCIS,x="number.of.lym.in.H.E.DCIS",y="number.of.foxp3.cells",add="reg.line",conf.int=TRUE,cor.coef = TRUE,cor.method = "spearman", xlab="HE", ylab="FoxP3")
ggarrange(p1, p2, p3, nrow = 1)
dev.off()


#a = tab[, c(2:4, 7)]


p = c(0.001, 0.5)
p.adjust(p, method = "BH")
