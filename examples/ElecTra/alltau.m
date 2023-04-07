%Developed by Emad
clear all
clc
load('TE_Si_kScan_electrons.mat')
disp('This is to save relaxation times');
%number of temperature and energy points
disp(tauE_sep(1,1))
emech=input('please enter number of mechanisms printed above');
tp=size(T_array,2);
Ep=size(E_array,2);
kdir=3;
etau=zeros(emech,tp,Ep);
%independent of chemical potnetial 
for itp=1:tp
    for iep=1:Ep
        etau(1,itp,iep)=(tauE_sep.ADP.x(iep,itp)+tauE_sep.ADP.y(iep,itp)+tauE_sep.ADP.z(iep,itp))/3;
        etau(2,itp,iep)=(tauE_sep.ODP.x(iep,itp)+tauE_sep.ODP.y(iep,itp)+tauE_sep.ODP.z(iep,itp))/3;
        etau(3,itp,iep)=(tauE_sep.POP.x(iep,itp)+tauE_sep.POP.y(iep,itp)+tauE_sep.POP.z(iep,itp))/3;
        etau(4,itp,iep)=(tauE_sep.IVS.x(iep,itp)+tauE_sep.IVS.y(iep,itp)+tauE_sep.IVS.z(iep,itp))/3;
        etau(5,itp,iep)=(tauE_sep.Alloy.x(iep,itp)+tauE_sep.Alloy.y(iep,itp)+tauE_sep.Alloy.z(iep,itp))/3;
    end 
end
dmrt=zeros(Ep,tp);
for itp=1:tp
    for iep=1:Ep
        for imch=1:emech
            if etau(imch,itp,iep)~=0.00 & isnan(etau(imch,itp,iep))==false
                dmrt(iep,itp)=dmrt(iep,itp)+1/etau(imch,itp,iep);
            end
        end
    end
end
%chemical potential dependent
mupt=size(EF_matrix,1);
murt=zeros(mupt,Ep,tp);
size(tauE_IIS.x);
for imu=1:mupt
    for itp=1:tp
        for iep=1:Ep
            if isnan(tauE_IIS.x(iep,imu,itp))==false & tauE_IIS.x(iep,imu,itp)~=0.00
                murt(imu,iep,itp)=1/tauE_IIS.x(iep,imu,itp);
            end
        end
    end
end
%total scattering rates         
fid=fopen('elctrates.in','w');
for itp=1:tp
    for imu=1:mupt
        fprintf(fid,'%12s %6.2f %8.4f \n','#T and mu',T_array(itp),EF_matrix(imu,itp));
        fprintf(fid,'%6s %12s\n','#E(eV)','rates(1/s)');
        for iep=1:Ep
            fprintf(fid,'%8.4f %e\n',E_array(1,iep),murt(imu,iep,itp)+dmrt(iep,itp));
        end
    end
end



