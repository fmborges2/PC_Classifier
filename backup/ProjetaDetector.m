%PROJETO DO DETECTOR

close all, clear all, clc
atraso = 1;

load Xf_t
load Xs_p
load Xs_t
load curva

nominal = Xs_t;
disturbios = Xf_t;


%Distancia 
[y,dd]=map_to_arcl(e,v,disturbios);
[y,dn]=map_to_arcl(e,v,nominal);
% 
plot(dd,'o'), hold, plot(dn,'r*')

dado = 320;
[pdf bins] = hist([dn' dd'],dado);
patamar = [min([dn']) bins];

for j=1:dado+1
    pa(j)=0;
    pf(j)=0;
    for i = 1:320
        if dd(i)>patamar(j)
            pa(j) = pa(j)+1;
        end
    end
    for i = 1:120
        if dn(i)>patamar(j)
            pf(j) = pf(j)+1;
        end
    end
end


PD = pa/320;
PF = pf/120;

figure 
plot(PF,PD,'-')
xlabel('PF')
ylabel('PD')

%save ROC13seg PF PD

figure
plot(patamar,PD,'-')
hold 
plot(patamar,PF,'-r')
grid
xlabel('Limiar')
ylabel('PD e PF')
legend('PD','PF')

[tpr,fpr,thresholds] = roc([ones(320,1); zeros(120,1)],[dd;dn]);
plotroc(fpr,tpr)