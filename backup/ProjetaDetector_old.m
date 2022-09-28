%PROJETO DO DETECTOR

close all, clear all, clc
atraso = 1;

Np = 250; %eventos de projeto

load InterrupcaoCurta25db
for k=1:Np
    aux1(k,:) = interrupcao_curta(k,no(k)-atraso:no(k)-atraso+255);
end
disturbios = aux1;

load Interrupcao25db
for k=1:Np
    aux2(k,:) = interrupcao(k,no(k)-atraso:no(k)-atraso+255);
end
disturbios = [disturbios; aux2];

load Sag25db
for k=1:Np
    aux3(k,:) = sag(k,no(k)-atraso:no(k)-atraso+255);
end
disturbios = [disturbios; aux3];

load Swell25db
for k=1:Np
    aux4(k,:) = swell(k,no(k)-atraso:no(k)-atraso+255);
end
disturbios = [disturbios; aux4];

load Spike25db
for k=1:Np
    aux5(k,:) = spike(k,no(k)-atraso:no(k)-atraso+255);
end
disturbios = [disturbios; aux5];

load Notch25db
for k=1:Np
    aux6(k,:) = notch(k,no(k)-atraso:no(k)-atraso+255);
end
disturbios = [disturbios; aux6];

load Har25db
for k=1:Np
    aux7(k,:) = har(k,no(k)-atraso:no(k)-atraso+255);
end
disturbios = [disturbios; aux7];

load Capc25db
for k=1:Np
    aux8(k,:) = capc(k,no(k)-atraso:no(k)-atraso+255);
end
disturbios = [disturbios; aux8];

load Norm25db
for k = 1:1000
    nominal(k,:) = norm(k,1:256);
end

load curva_norm4seg
%Distancia 
[y,dd]=map_to_arcl(edges_norm,vertices_norm,disturbios);
[y,dn]=map_to_arcl(edges_norm,vertices_norm,nominal);
% 
plot(dd,'o'), hold, plot(dn,'r*')

%nb = NaiveBayes.fit([dd; dn],[ones(2000,1); zeros(1000,1)]);
%save bayes30dbPC nb 

%Limiar = 0,75 

% [y,d]=map_to_arcl(edges_norm,vertices_norm,aux4);
% 
% 
% min(d)

dado = 2000;
[pdf bins] = hist([dn' dd'],dado);
patamar = [min([dn']) bins];

for j=1:dado+1
    pa(j)=0;
    pf(j)=0;
    for i = 1:2000
        if dd(i)>patamar(j)
            pa(j) = pa(j)+1;
        end
    end
    for i = 1:1000
        if dn(i)>patamar(j)
            pf(j) = pf(j)+1;
        end
    end
end


PD = pa/2000;
PF = pf/1000;

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

[tpr,fpr,thresholds] = roc([ones(2000,1); zeros(1000,1)],[dd;dn]);
plotroc(fpr,tpr)