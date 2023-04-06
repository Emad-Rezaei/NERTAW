# Python program to demonstrate
# HDF5 file
import numpy as np
from numpy.linalg import inv, det
import h5py
import math
import time
import matplotlib.pyplot as plt
from pymatgen.electronic_structure.core import Spin, Orbital
from amset.io import load_mesh

start = time.time()

print('''
$   $  $$$$   $$$$   $$$$$       $       $    $    $
$$  $  $      $  $     $        $ $      $   $ $   $
$ $ $  $$$$   $$$$     $       $$$$$     $  $   $  $
$  $$  $      $$       $      $     $    $ $     $ $
$   $  $$$$   $ $      $     $       $   $$       $$
''')

print('''please cite the following paper:
Calculation of thermomagnetic properties using first-principles density functional theory,
Rezaei, S Emad and Zebarjadi, Mona and Esfarjani, Keivan,
Computational Materials Science
''')
#Default parameters
T=300.00
Evwan=0.00
Ev=0.00
Eg=1.0
x=0.1
mustep=0.01
Bex=1.00
nerbex=1.00
bni=0
bnf=1
eps0=1e-5
calcdos=0
calcseeb=0
savematrix=0
amesh=0
Tmin=300.0
Tmax=301.00
Tstep=1
ctau=1.0
#reading input parameters
with open('inpner.in', 'r') as f:
     input_data = f.readlines()
inppar = []
for p in input_data:
    inppar.append(p.split(" "))
f.close()
inppar= np.array(inppar)
for i in range(0,inppar.shape[0]):
    if inppar[i,0]=='smrT':
        T=float(inppar[i,1])
    if inppar[i,0]=='Evwan':
        Evwan=float(inppar[i,1])
    if inppar[i,0]=='Ev':
        Ev=float(inppar[i,1])
    if inppar[i,0]=='Eg':
        Eg=float(inppar[i,1])
    if inppar[i,0]=='mubn':
        x=float(inppar[i,1])
    if inppar[i,0]=='mustep':
        mustep=float(inppar[i,1])
    if inppar[i,0]=='prefix':
        prefix=str(inppar[i,1])
    if inppar[i,0]=='ratecode':
        ratecode=str(inppar[i,1])
    if inppar[i,0]=='elctcharge':
        elctcharge=str(inppar[i,1])
    if inppar[i,0]=='Bex':
        Bex=float(inppar[i,1])
    if inppar[i,0]=='ner_bext':
        nerbex=float(inppar[i,1])
    if inppar[i,0]=='Tmin':
        Tmin=float(inppar[i,1])
    if inppar[i,0]=='Tmax':
        Tmax=float(inppar[i,1])
    if inppar[i,0]=='ctau':
        ctau=float(inppar[i,1])
    if inppar[i,0]=='Tstep':
        Tstep=float(inppar[i,1])
    if inppar[i,0]=='amesh':
        amesh=int(inppar[i,1])
    if inppar[i,0]=='bni':
        bni=int(inppar[i,1])
    if inppar[i,0]=='bnf':
        bnf=int(inppar[i,1])
    if inppar[i,0]=='eps0':
        eps0=float(inppar[i,1])
    if inppar[i,0]=='calcdos':
        calcdos=int(inppar[i,1])
    if inppar[i,0]=='calcseeb':
        calcseeb=int(inppar[i,1])
    if inppar[i,0]=='savematrix':
        savematrix=int(inppar[i,1])


#writing in log file
print(f'smearing temperature: {T} K')
print(f'Top of wannierized valence band: {Evwan} ev')
print(f'Top of valence band in the scattering rate calculations: {Ev} ev')
print(f'chemical potential in each band: {x} ev')
print(f'chemical potnetial step size: {mustep} eV')
print(f'prefix of the calculations: {prefix}')
print(f'The code used to compute scattering rate: {ratecode}')
print(f'External magnetic field applied in tdf files from the Wannier90: {nerbex} T')
print(f'External magnetic field applied here to obtain the Nernst coefficient: {Bex} T')
if ratecode[0:len(ratecode)-1].upper()=='AMSET':
    print(f'mesh in the AMSET code: {amesh}')
print(f'whether dos is calculated: {calcdos}')
if bool(calcdos)==True:
    print(f'Initial band index for DOS: {bni}')
    print(f'Final band index for DOS: {bnf}')
print(f'tolerance for Fermi-dirac function: {eps0}')
if ratecode[0:len(ratecode)-1].upper()=='CRTA':
    print(f'The constant relaxation time approximation is assumed')
    print(f'Maximum temperature: {Tmin} K')
    print(f'Maximum temperature: {Tmax} K')
    print(f'Temperature step size: {Tstep}')
    print(f'Constant value for the relaxation time: {ctau} fs')
print(f'whether the Seebeck coefficient is calculated: {calcseeb}')
print(f'whether coefficient matrices are saved: {savematrix}')
print('''Coefficients are explained in:
Calculation of thermomagnetic properties using first-principles density functional theory,
Rezaei, S Emad and Zebarjadi, Mona and Esfarjani, Keivan,
Computational Materials Science
''')




#Functions
#Fermi-Dirac
def ferdir(chem,enr,tp,Ke,ep):
    if enr <=chem+Ke*tp*math.log(ep):
        fn=1
        dv=0
    elif enr >= chem+Ke*tp*(-1)*math.log(ep):
        fn=0
        dv=0
    else:
        fn=1/(1+math.exp((enr-chem)/Ke/tp))
        dv = fn*(1-fn)/(Ke*tp)
    return fn,dv

#Reading Perturbo data
def pertread(prefix,eigstep,eigint,eigpt,Ec,Ev,x,T,kb,eps0,q,h):
    perfil="{}.imsigma".format(prefix[0:len(prefix)-1])
    pertmp="{}.temper".format(prefix[0:len(prefix)-1])
    per=np.loadtxt(perfil)
    temper=np.loadtxt(pertmp,skiprows=1)
    nbnd=0
    nk=0
    ik=0
    it=0
    while per[1,ik]==per[1,0]:
       nbnd+=1
       ik+=1
    print(f"number of bands: {nbnd}")
    while per[it,0]==per[0,0]:
       nk+=1
       it+=1
    nk=int(nk/nbnd)
    print(f"number of kpoints: {nk}")
    dop=0
    for i in range(0,temper.shape[0]):
        if temper[i,0]==temper[0,0]:
            dop+=1
    tmp=int(per.shape[0]/nk/nbnd/dop)
    Dop=np.zeros((dop * tmp, tmp))
    Temp=np.zeros(tmp)
    print(f"number of doping points: {dop}")
    print(f"number of temperature points: {tmp}")
    for idp in range(0,dop*tmp):
        Dop[idp,:]=temper[idp,2]
    print("temperature values are:")
    for itmp in range(0,tmp):
        Temp[itmp]=temper[dop*itmp,0]
    print(Temp)
    print("Doping values are:")
    for itmp in range(0,tmp):
        for idp in range(0,dop):
            print(Dop[idp + tmp * itmp, itmp])
    tau=np.zeros((eigpt,dop,tmp))
    sdfde=np.zeros((eigpt,1)) #sum of delta functio
    for ieig in range(0,eigpt):
        eig[ieig]=eigint+eigstep*ieig
        if eig[ieig] >= Ev-2*x and eig[ieig] <= Ec+2*x:
            for itmp in range(0,tmp):
                for idp in range(0,dop):
                    for iper in range((idp+itmp*tmp)*nbnd*nk,(idp+itmp*tmp+1)*nbnd*nk):
                        fd,dfde=ferdir(eig[ieig],per[iper,3],T,kb,eps0)
                        tau[ieig,idp,itmp]+=per[iper,4]*dfde
                        sdfde[ieig,0]+=dfde
                    tau[ieig,idp,itmp]/=sdfde[ieig,0]
    tau*=(q/h)
#shifting to the top of valence band
    for ieig in range(0,eigpt):
        eig[ieig]-=Ev
#Total relaxation time by inversing
    for idp in range(0,dop):
        for itmp in range(0,tmp):
            for ieig in range(0,eigpt):
                if math.isnan(tau[ieig,idp,itmp])==False:
                    if tau[ieig,idp,itmp]!=0.00:
                        tau[ieig,idp,itmp]=1/tau[ieig,idp,itmp]
                else:
                    tau[ieig,idp,itmp]=0.00

    for itmp in range(0,tmp):
        for idp in range(0,dop):
            mta="T{}dop{:.3e}tau.txt".format(str(Temp[itmp]),Dop[idp+tmp*itmp, itmp])
            with open(mta,'w') as f:
                f.writelines('energy(eV)    tau(s)' + '\n')
                np.savetxt(f,np.c_[eig,tau[:,idp,itmp]], delimiter=' ')

    return tau,dop,tmp,Temp,Dop
#Reading AMSET data
def amsread (amesh,eigstep,eigint,eigpt,Ec,Ev,x,T,kb,eps0):
#loading data
    meshdat="mesh_{}x{}x{}.h5".format(amesh,amesh,amesh)
    data = load_mesh(meshdat)
#number of bands
    eng=data["energies"]
    nbnd=eng[Spin.up].shape[0]
    print(f'number of bands: {nbnd}')
    nk=eng[Spin.up].shape[1]
    print(f'number of kpoints: {nk}')
#scattering rates
    rates = data["scattering_rates"]
#scattering labels
    mechs=data["scattering_labels"]
#dopings
    dmDop=data["doping"]
#point number of mechanism,doping, temperature,bands,kpoints(4, 1, 2, 5, 969)
    mech=rates[Spin.up].shape[0]
    dop=rates[Spin.up].shape[1]
    tmp=rates[Spin.up].shape[2]
    Dop = np.zeros((dop, tmp))
    for idp in range(0, dop):
        Dop[idp, :] = dmDop[idp]
    Temp=np.zeros(tmp)
    Temp=data["temperatures"]
    print(f'number of doping points:{dop}')
    print(f'number of temperature points:{tmp}')
#scattering rates vs energy
    srt=np.zeros((eigpt,mech,dop,tmp))
    sdfde=np.zeros((eigpt,1)) #sum of delta function
    dm=np.zeros((eigpt,mech))

#Chemical potential and fermi-dirac
#derivative of f
    for ieig in range(0,eigpt):
        eig[ieig]=eigint+eigstep*ieig
        if eig[ieig] >= Ev-2*x and eig[ieig] <= Ec+2*x:
            for imch in range(0,mech):
                for idp in range(0,dop):
                    for itmp in range(0,tmp):
                        #for i in [0,1,2,3,4]:
                        for ibn in range(0,nbnd):
                            for ik in range(0,nk):
                                fd,dfde=ferdir(eig[ieig],eng[Spin.up][ibn,ik],T,kb,eps0)
                                srt[ieig,imch,idp,itmp]+=rates[Spin.up][imch,idp,itmp,ibn,ik]*dfde
                                sdfde[ieig,0]+=dfde
                        srt[ieig,imch,idp,itmp]/=sdfde[ieig,0]
#shifting to the top of valence band
    for ieig in range(0,eigpt):
        eig[ieig]-=Ev
#Total scattering rates
    tau=np.zeros((eigpt,dop,tmp))
    for idp in range(0,dop):
        for itmp in range(0,tmp):
            for imch in range(0,mech):
                tau[:,idp,itmp]+=srt[:,imch,idp,itmp]

    for itmp in range(0,tmp):
        for idp in range(0,dop):
            mta="T{}dop{:.3e}rate.txt".format(str(Temp[itmp]),Dop[idp,itmp])
            for imch in range(0,mech):
                for ieig in range(0,eigpt):
                    if math.isnan(srt[ieig,imch,idp,itmp])==False:
                        dm[ieig,imch]=srt[ieig,imch,idp,itmp]
                    else:
                        dm[ieig,imch]=0.00
            with open(mta,'w') as f:
                f.writelines('#Scattering rates in the units of s' +'\n')
                f.writelines('#Energy values are shifted to the top of valence band' +'\n')
                f.writelines('#energy(eV)' +str(mechs)+ '\n')
                np.savetxt(f,np.c_[eig,dm], delimiter=' ')
#Total relaxation time by inversing
    for idp in range(0,dop):
        for itmp in range(0,tmp):
            for ieig in range(0,eigpt):
                if math.isnan(tau[ieig,idp,itmp])==False:
                    if tau[ieig,idp,itmp]!=0.00:
                        tau[ieig,idp,itmp]=1/tau[ieig,idp,itmp]
                else:
                    tau[ieig,idp,itmp]=0.00

    return tau,dop,tmp,Temp,Dop
#Reading ElecTra data
def elctread(elctcharge,eigstep,eigpt,Evwan,Eg,tdf1):
    elc=np.loadtxt('elcrates.in')
    Temp=np.loadtxt('elcT.in')
    Dop=np.loadtxt("elcmu.in")
    tmp=Temp.shape[0]
    dop=Dop.shape[0]
#number of energy points
    for i in range(1,elc.shape[0]):
        if elc[i,0]==elc[0,0]:
            break
    ept=i
    ear=np.zeros(math.floor(1+(elc[ept-1,0]-elc[0,0])/eigstep))
    dmrt=np.zeros((tmp,dop,ear.shape[0]))
#energy array
    for i in range(0,ear.shape[0]):
        ear[i]=elc[0,0]+eigstep*i
#interpolation
    for itp in range(0,tmp):
        for imu in range(0,dop):
            for ieg in range(0,dmrt.shape[2]):
                mn=1e4
                mni=0
                for iep in range(0,ept):
                    if abs(ear[ieg]-elc[iep,0])<eigstep/2:
                        dmrt[itp,imu,ieg]=abs(elc[(itp*dop+imu)*ept+iep,1])
                    else:
                        if abs(ear[ieg]-elc[iep,0])<mn:
                            mn=abs(ear[ieg]-elc[iep,0])
                            mni=iep+(itp*dop+imu)*ept
                            if mni <tmp*dop*ept-1:
                                dmrt[itp,imu,ieg]=abs(elc[mni,1]+(elc[mni+1,1]-elc[mni-1,1])/(elc[mni+1,0]-elc[mni-1,0])*mn)
#shifting to the top of valence band 
    Esh= 0
    if elctcharge[0:len(elctcharge)-1].upper()=='E':
        Esh=Eg
    print(f"elctcharge : {elctcharge}")
    print(f"Esh: {Esh}")
    print(f"Eg: {Eg}")
#expanding to TDF size
    tau=np.zeros((eigpt,dop,tmp))
    for itp in range(0,tmp):
        for imu in range(0,dop):
            for ieng in range(0,eigpt):
                if tdf1[ieng,0]-Evwan >= ear[0]+Esh-10*eigstep and tdf1[ieng,0]-Evwan <= ear[ear.shape[0]-1]+Esh+10*eigstep:
                    for ier in range(0,ear.shape[0]):
                        if abs(tdf1[ieng,0]-Evwan-ear[ier]-Esh)<eigstep:
                            if dmrt[itp,imu,ier]!=0.00:
                                tau[ieng,imu,itp]=1/dmrt[itp,imu,ier]
    for itmp in range(0,tmp):
        for imu in range(0,dop):
            mta="T{}dop{:.3e}tau.txt".format(str(Temp[itmp]),Dop[imu,itmp])
            with open(mta,'w') as f:
                f.writelines('#Zero is the top of the valence band' + '\n')
                f.writelines('#energy(eV)    tau(s)' + '\n')
                np.savetxt(f,np.c_[tdf1[:,0]-Evwan,tau[:,imu,itmp]], delimiter=' ')

    return tau,dop,tmp,Temp,Dop

def customread(eigstep,eigpt,Ev,Evwan,Eg,tdf1):
    cusrate = np.loadtxt('rarray.in')
    Temp = np.loadtxt('Tarray.in')
    Dop = np.loadtxt('muarray.in')
    dop = Dop.shape[0]
    tmp = Temp.shape[0]
    # number of energy points
    ept = cusrate.shape[0]
    ear = np.zeros(math.floor(1 + (cusrate[ept - 1, 0] - cusrate[0, 0]) / eigstep))
    dmrt = np.zeros((tmp, dop, ear.shape[0]))
    # energy array
    for i in range(0, ear.shape[0]):
        ear[i] = cusrate[0, 0] + eigstep * i
    # interpolation
    for itp in range(0, tmp):
        for imu in range(0, dop):
            for ieg in range(0, dmrt.shape[2]):
                mn = 1e4
                for iep in range(0, ept-1):
                    if abs(ear[ieg] - cusrate[iep, 0]) < eigstep / 2:
                        dmrt[itp, imu, ieg] = abs(cusrate[iep, 1 + itp * dop + imu])
                    else:
                        if abs(ear[ieg] - cusrate[iep, 0]) < mn:
                            mn = abs(ear[ieg] - cusrate[iep, 0])
                            dmrt[itp, imu, ieg] = abs(cusrate[iep, 1 + itp * dop + imu] +
                                                      (cusrate[iep + 1, 1 + itp * dop + imu] -
                                                       cusrate[iep - 1, 1 + itp * dop + imu]) /
                                                      (cusrate[iep + 1, 0] - cusrate[iep - 1, 0]) * mn)
    # shifting to the top of valence band
    # expanding to TDF size
    tau = np.zeros((eigpt, dop, tmp))
    for itp in range(0, tmp):
        for imu in range(0, dop):
            for ieng in range(0, eigpt):
                if ear[0] - Ev - 10 * eigstep <= tdf1[ieng, 0] - Evwan <= \
                        ear[ear. shape[0] - 1] + Ev + 10 * eigstep:
                    for ier in range(0, ear.shape[0]):
                        if abs(tdf1[ieng, 0] - Evwan - ear[ier] + Ev) < eigstep:
                            if dmrt[itp, imu, ier] != 0.00:
                                tau[ieng, imu, itp] = 1 / dmrt[itp, imu, ier]
    for itmp in range(0, tmp):
        for imu in range(0, dop):
            mta = "T{}dop{:.3e}tau.txt".format(str(Temp[itmp]), Dop[imu, itmp])
            with open(mta, 'w') as f:
                f.writelines('#Zero is the top of the valence band' + '\n')
                f.writelines('energy- Ev (eV)    tau(s)' + '\n')
                np.savetxt(f, np.c_[tdf1[:, 0] - Evwan, tau[:, imu, itmp]], delimiter=' ')
    return tau,dop,tmp,Temp,Dop


#DOS
def amsdos (amesh,eigstep,eigint,eigpt,T,kb,eps0,bni,bnf,Ev):
#loading data
    meshdat="mesh_{}x{}x{}.h5".format(amesh,amesh,amesh)
    data = load_mesh(meshdat)
#number of bands
    eng=data["energies"]
    nbnd=eng[Spin.up].shape[0]
    nk=eng[Spin.up].shape[1]
    dos=np.zeros((eigpt,2)) #sum of delta function
    for ieig in range(0,eigpt):
        eig[ieig]=eigint+eigstep*ieig
        dos[ieig,0]=eig[ieig]-Ev
        #if eig[ieig] >= Ev-2*x and eig[ieig] <= Ec+2*x:
        for ibn in range(bni,bnf+1):
            for ik in range(0,nk):
                fd,dfde=ferdir(eig[ieig],eng[Spin.up][ibn,ik],T,kb,eps0)
                dos[ieig,1]+=dfde

    dos[:,1]/=(ibn*ik)
    mds="dos.txt"
    with open(mds,'w') as ds:
        ds.writelines('Density of States ' + '\n')
        ds.writelines('Zero is top of the valence band ' + '\n')
        ds.writelines('E(eV)	DOS(1/eV)' + '\n')
        np.savetxt(ds,dos, delimiter=' ')

#Total TDF
def SEEBTDF(tdf1,tau,Evwan,dop,tmp):
#shifting to the top of valence band
#total distribution function xx xy xz yx yy yz zx zy zz
    for idp in range(0,dop):
        for itmp in range(0,tmp):
            for i in range(0,tdf1.shape[0]):
                TDFseeb[i,0,idp,itmp]=tdf1[i,0]-Evwan
                TDFseeb[i,1,idp,itmp]=tdf1[i,1]*tau[i,idp,itmp]
                TDFseeb[i,2,idp,itmp]=tdf1[i,2]*tau[i,idp,itmp]
                TDFseeb[i,3,idp,itmp]=tdf1[i,4]*tau[i,idp,itmp]
                TDFseeb[i,4,idp,itmp]=tdf1[i,7]*tau[i,idp,itmp]
                TDFseeb[i,5,idp,itmp]=tdf1[i,3]*tau[i,idp,itmp]
                TDFseeb[i,6,idp,itmp]=tdf1[i,5]*tau[i,idp,itmp]
                TDFseeb[i,7,idp,itmp]=tdf1[i,8]*tau[i,idp,itmp]
                TDFseeb[i,8,idp,itmp]=tdf1[i,9]*tau[i,idp,itmp]
                TDFseeb[i,9,idp,itmp]=tdf1[i,6]*tau[i,idp,itmp]
    return TDFseeb

#Integration of response functions for the Seebeck effect
def intmatseeb(dop,tmp,mupt,ndir,eigpt,TDFseeb,mu,Temp,kb,eps0,sbsig,sbBmtr,sbkmtr,sbmsig,sbmB,sbmk,sbmrho,sbmS,kap):
    for idp in range(0,dop):
        for itmp in range(0,tmp):
            for imu in range(0,mupt):
                for idir in range(1,ndir+1):
                    for ieig in range(0,eigpt-1):
                        if abs(TDFseeb[ieig,0,idp,itmp]-mu[imu]) < 0.3:
                            fd1,mdfde1=ferdir(mu[imu],TDFseeb[ieig,0,idp,itmp],Temp[itmp],kb,eps0)
                            fd2,mdfde2=ferdir(mu[imu],TDFseeb[ieig+1,0,idp,itmp],Temp[itmp],kb,eps0)
                            sbsig[imu,idir,idp,itmp]+=0.5*eigstep*(TDFseeb[ieig+1,idir,idp,itmp]*mdfde2+TDFseeb[ieig,idir,idp,itmp]*mdfde1)
                            sbBmtr[imu,idir,idp,itmp]+=0.5*eigstep/(-1*Temp[itmp])*(TDFseeb[ieig+1,idir,idp,itmp]*mdfde2*(TDFseeb[ieig+1,0,idp,itmp]-mu[imu])+TDFseeb[ieig,idir,idp,itmp]*mdfde1*(TDFseeb[ieig,0,idp,itmp]-mu[imu]))
                            sbkmtr[imu,idir,idp,itmp]+=0.5*eigstep/(Temp[itmp])*(TDFseeb[ieig+1,idir,idp,itmp]*mdfde2*(TDFseeb[ieig+1,0,idp,itmp]-mu[imu])**2+TDFseeb[ieig,idir,idp,itmp]*mdfde1*(TDFseeb[ieig,0,idp,itmp]-mu[imu])**2)

            #conduscitivty 3*3 matrix 
                sbmsig=sbsig[imu, 1:, idp, itmp].reshape(3, 3)
            #B 3*3 matrix 
                sbmB=sbBmtr[imu,1:,idp,itmp].reshape(3,3)
            #K 3*3 matrix 
                sbmk[0,0]=sbkmtr[imu,1:,idp,itmp].reshape(3,3)
                if det(sbmsig)!=0.00:
                    sbmrho[imu,idp,itmp,:,:]=inv(sbmsig)
                    sbmS[imu,idp,itmp,:,:]=np.matmul(sbmrho[imu,idp,itmp,:,:],sbmB)
#electronic thermal conductivity
                kap[imu,idp,itmp,:,:]=sbmk-Temp[itmp]*np.matmul(sbmsig,np.matmul(sbmS[imu,idp,itmp,:,:],sbmS[imu,idp,itmp,:,:]))

    return sbsig,sbBmtr,sbkmtr,sbmS,kap


#Total TDF
def TOTTDF(tdf1,tdf2,tau,Bex,nerbex,Evwan,dop,tmp):
#shifting to the top of valence band
#total distribution function xx xy xz yx yy yz zx zy zz
    Bext=Bex/nerbex
    for idp in range(0,dop):
        for itmp in range(0,tmp):
            for i in range(0,tdf1.shape[0]):
                TDFtot[i,0,idp,itmp]=tdf1[i,0]-Evwan
                TDFtot[i,1,idp,itmp]=tdf1[i,1]*tau[i,idp,itmp]+tdf2[i,1]*Bext*tau[i,idp,itmp]**2
                TDFtot[i,2,idp,itmp]=tdf1[i,2]*tau[i,idp,itmp]+tdf2[i,2]*Bext*tau[i,idp,itmp]**2
                TDFtot[i,3,idp,itmp]=tdf1[i,4]*tau[i,idp,itmp]+tdf2[i,4]*Bext*tau[i,idp,itmp]**2
                TDFtot[i,4,idp,itmp]=tdf1[i,7]*tau[i,idp,itmp]+tdf2[i,7]*Bext*tau[i,idp,itmp]**2
                TDFtot[i,5,idp,itmp]=tdf1[i,3]*tau[i,idp,itmp]+tdf2[i,3]*Bext*tau[i,idp,itmp]**2
                TDFtot[i,6,idp,itmp]=tdf1[i,5]*tau[i,idp,itmp]+tdf2[i,5]*Bext*tau[i,idp,itmp]**2
                TDFtot[i,7,idp,itmp]=tdf1[i,8]*tau[i,idp,itmp]+tdf2[i,8]*Bext*tau[i,idp,itmp]**2
                TDFtot[i,8,idp,itmp]=tdf1[i,9]*tau[i,idp,itmp]+tdf2[i,9]*Bext*tau[i,idp,itmp]**2
                TDFtot[i,9,idp,itmp]=tdf1[i,6]*tau[i,idp,itmp]+tdf2[i,6]*Bext*tau[i,idp,itmp]**2
    return TDFtot,Bext
#Integration of response functions for the Nernst effect
def intmat(dop,tmp,mupt,ndir,eigpt,TDFtot,mu,Temp,kb,eps0,sig,Bmtr,kmtr,msig,mB,mk,mrho,mS,kappa):

    for idp in range(0,dop):
        for itmp in range(0,tmp):
            for imu in range(0,mupt):
                for idir in range(1,ndir+1):
                    for ieig in range(0,eigpt-1):
                        if abs(TDFtot[ieig,0,idp,itmp]-mu[imu]) < 0.3:
                            fd1,mdfde1=ferdir(mu[imu],TDFtot[ieig,0,idp,itmp],Temp[itmp],kb,eps0)
                            fd2,mdfde2=ferdir(mu[imu],TDFtot[ieig+1,0,idp,itmp],Temp[itmp],kb,eps0)
                            sig[imu,idir,idp,itmp]+=0.5*eigstep*(TDFtot[ieig+1,idir,idp,itmp]*mdfde2+TDFtot[ieig,idir,idp,itmp]*mdfde1)
                            Bmtr[imu,idir,idp,itmp]+=0.5*eigstep/(-1*Temp[itmp])*(TDFtot[ieig+1,idir,idp,itmp]*mdfde2*(TDFtot[ieig+1,0,idp,itmp]-mu[imu])+TDFtot[ieig,idir,idp,itmp]*mdfde1*(TDFtot[ieig,0,idp,itmp]-mu[imu]))
                            kmtr[imu,idir,idp,itmp]+=0.5*eigstep/(Temp[itmp])*(TDFtot[ieig+1,idir,idp,itmp]*mdfde2*(TDFtot[ieig+1,0,idp,itmp]-mu[imu])**2+TDFtot[ieig,idir,idp,itmp]*mdfde1*(TDFtot[ieig,0,idp,itmp]-mu[imu])**2)
            #conduscitivty 3*3 matrix 
                msig=sig[imu, 1:, idp, itmp].reshape(3, 3)
            #B 3*3 matrix 
                mB=Bmtr[imu,1:,idp,itmp].reshape(3,3)
            #K 3*3 matrix 
                mk=kmtr[imu,1:,idp,itmp].reshape(3,3)
                if det(msig)!=0.00:
                    mrho[imu,idp,itmp,:,:]=inv(msig)
                    mS[imu,idp,itmp,:,:]=np.matmul(mrho[imu,idp,itmp,:,:],mB)
#electronic thermal conductivity
                kappa[imu,idp,itmp,:,:]=mk-Temp[itmp]*np.matmul(msig,np.matmul(mS[imu,idp,itmp,:,:],mS[imu,idp,itmp,:,:]))

    return sig,mrho,Bmtr,kmtr,mS,kappa

def CTSEEBTDF(tdf1,Evwan,ctau):
#shifting to the top of valence band
#total distribution function xx xy xz yx yy yz zx zy zz
    for i in range(0,tdf1.shape[0]):
        TDFseeb[i,0]=tdf1[i,0]-Evwan
        TDFtot[i,1]=tdf1[i,1]*ctau*1e-15
        TDFtot[i,2]=tdf1[i,2]*ctau*1e-15
        TDFtot[i,3]=tdf1[i,4]*ctau*1e-15
        TDFtot[i,4]=tdf1[i,7]*ctau*1e-15
        TDFtot[i,5]=tdf1[i,3]*ctau*1e-15
        TDFtot[i,6]=tdf1[i,5]*ctau*1e-15
        TDFtot[i,7]=tdf1[i,8]*ctau*1e-15
        TDFtot[i,8]=tdf1[i,9]*ctau*1e-15
        TDFtot[i,9]=tdf1[i,6]*ctau*1e-15
    return TDFseeb
#Temperature array
def TempAr(Tmin,Tmax,Tstep):
    tmp=math.floor((Tmax-Tmin)/Tstep)+1
    Temp=np.zeros(tmp)
    for itmp in range(0,tmp):
        Temp[itmp]=Tmin+itmp*Tstep
    return tmp,Temp
#Integration of response functions for the Seebeck effect
def CTintmatseeb(tmp,mupt,ndir,eigpt,TDFseeb,mu,Temp,kb,eps0,sbsig,sbBmtr,sbkmtr,sbmsig,sbmB,sbmk,sbmrho,sbmS,kap):
#Temperature array
    for itmp in range(0,tmp):
        for imu in range(0,mupt):
            for idir in range(1,ndir+1):
                for ieig in range(0,eigpt-1):
                    if abs(TDFseeb[ieig,0]-mu[imu]) < 0.3:
                        fd1,mdfde1=ferdir(mu[imu],TDFseeb[ieig,0],Temp[itmp],kb,eps0)
                        fd2,mdfde2=ferdir(mu[imu],TDFseeb[ieig+1,0],Temp[itmp],kb,eps0)
                        sbsig[imu,idir,itmp]+=0.5*eigstep*(TDFseeb[ieig+1,idir]*mdfde2+TDFseeb[ieig,idir]*mdfde1)
                        sbBmtr[imu,idir,itmp]+=0.5*eigstep/(-1*Temp[itmp])*(TDFseeb[ieig+1,idir]*mdfde2*(TDFseeb[ieig+1,0]-mu[imu])+TDFseeb[ieig,idir]*mdfde1*(TDFseeb[ieig,0]-mu[imu]))
                        sbkmtr[imu,idir,itmp]+=0.5*eigstep/(Temp[itmp])*(TDFseeb[ieig+1,idir]*mdfde2*(TDFseeb[ieig+1,0]-mu[imu])**2+TDFseeb[ieig,idir]*mdfde1*(TDFseeb[ieig,0]-mu[imu])**2)
            #conduscitivty 3*3 matrix
            sbmsig=sbsig[imu,1:,itmp].reshape(3,3)
            #B 3*3 matrix
            sbmB=sbBmtr[imu,1:,itmp].reshape(3,3)
            #K 3*3 matrix
            sbmk=sbkmtr[imu,1:,itmp].reshape(3,3)
            if det(sbmsig)!=0.00:
                sbmrho[imu,itmp,:,:]=inv(sbmsig)
                sbmS[imu,itmp,:,:]=np.matmul(sbmrho[imu,itmp,:,:],sbmB)
#electronic thermal conductivity
            kap[imu,itmp,:,:]=sbmk-Temp[itmp]*np.matmul(sbmsig,np.matmul(sbmS[imu,itmp,:,:],sbmS[imu,itmp,:,:]))

    return sbsig,sbBmtr,sbkmtr,sbmS,kap


#Total TDF
def CTTOTTDF(tdf1,tdf2,ctau,Bex,nerbex,Evwan):
#shifting to the top of valence band
#total distribution function xx xy xz yx yy yz zx zy zz
    Bext=Bex/nerbex
    for i in range(0,tdf1.shape[0]):
        TDFtot[i,0]=tdf1[i,0]-Evwan
        TDFtot[i,1]=tdf1[i,1]*ctau*1e-15+tdf2[i,1]*Bext*ctau**2*1e-30
        TDFtot[i,2]=tdf1[i,2]*ctau*1e-15+tdf2[i,2]*Bext*ctau**2*1e-30
        TDFtot[i,3]=tdf1[i,4]*ctau*1e-15+tdf2[i,4]*Bext*ctau**2*1e-30
        TDFtot[i,4]=tdf1[i,7]*ctau*1e-15+tdf2[i,7]*Bext*ctau**2*1e-30
        TDFtot[i,5]=tdf1[i,3]*ctau*1e-15+tdf2[i,3]*Bext*ctau**2*1e-30
        TDFtot[i,6]=tdf1[i,5]*ctau*1e-15+tdf2[i,5]*Bext*ctau**2*1e-30
        TDFtot[i,7]=tdf1[i,8]*ctau*1e-15+tdf2[i,8]*Bext*ctau**2*1e-30
        TDFtot[i,8]=tdf1[i,9]*ctau*1e-15+tdf2[i,9]*Bext*ctau**2*1e-30
        TDFtot[i,9]=tdf1[i,6]*ctau*1e-15+tdf2[i,6]*Bext*ctau**2*1e-30
    return TDFtot,Bext
#Integration of response functions for the Nernst effect
def CTintmat(tmp,mupt,ndir,eigpt,TDFtot,mu,Temp,kb,eps0,sig,Bmtr,kmtr,msig,mB,mk,mrho,mS,kappa):
    for itmp in range(0,tmp):
        for imu in range(0,mupt):
            for idir in range(1,ndir+1):
                for ieig in range(0,eigpt-1):
                    if abs(TDFtot[ieig,0]-mu[imu]) < 0.3:
                        fd1,mdfde1=ferdir(mu[imu],TDFtot[ieig,0],Temp[itmp],kb,eps0)
                        fd2,mdfde2=ferdir(mu[imu],TDFtot[ieig+1,0],Temp[itmp],kb,eps0)
                        sig[imu,idir,itmp]+=0.5*eigstep*(TDFtot[ieig+1,idir]*mdfde2+TDFtot[ieig,idir]*mdfde1)
                        Bmtr[imu,idir,itmp]+=0.5*eigstep/(-1*Temp[itmp])*(TDFtot[ieig+1,idir]*mdfde2*(TDFtot[ieig+1,0]-mu[imu])+TDFtot[ieig,idir]*mdfde1*(TDFtot[ieig,0]-mu[imu]))
                        kmtr[imu,idir,itmp]+=0.5*eigstep/(Temp[itmp])*(TDFtot[ieig+1,idir]*mdfde2*(TDFtot[ieig+1,0]-mu[imu])**2+TDFtot[ieig,idir]*mdfde1*(TDFtot[ieig,0]-mu[imu])**2)
            #conduscitivty 3*3 matrix
            msig=sig[imu,1:,itmp].reshape(3,3)
            #B 3*3 matrix
            mB=Bmtr[imu,1:,itmp].reshape(3,3)
            #K 3*3 matrix
            mk=kmtr[imu,1:,itmp].reshape(3,3)
            if det(msig)!=0.00:
                mrho[imu,itmp,:,:]=inv(msig)
                mS[imu,itmp,:,:]=np.matmul(mrho[imu,itmp,:,:],mB)
#electronic thermal conductivity
            kappa[imu,itmp,:,:]=mk-Temp[itmp]*np.matmul(msig,np.matmul(mS[imu,itmp,:,:],mS[imu,itmp,:,:]))

    return sig,mrho,Bmtr,kmtr,mS,kappa

#chemical potential 
def ChempotAr(mupt,x):
    for i in range(0,mupt):
        mu[i]=-x+mustep*i
    return mu


#constants
kb=8.617e-5 #ev
q=1.6e-19
h=6.626*1e-34 
#Reading TDFs from NERWAN
tf1="{}_tdf1.dat".format(prefix[0:len(prefix)-1])
tf2="{}_tdf2.dat".format(prefix[0:len(prefix)-1])
tdf1=np.loadtxt(tf1)
tdf2=np.loadtxt(tf2)
eigint=Ev+tdf1[0,0]-Evwan
eigstep=tdf1[1,0]-tdf1[0,0]
eigpt=tdf1.shape[0]
Ec=Ev+Eg
eig=np.zeros(eigpt)	#chemical potential
#Calling relaxation time
ratecode=ratecode[0:len(ratecode)-1].upper()
print(f'ratecode is {ratecode}')
if ratecode=='AMSET':
    tau,dop,tmp,Temp,Dop=amsread(amesh,eigstep,eigint,eigpt,Ec,Ev,x,T,kb,eps0)
elif ratecode=='PERTURBO':
    tau,dop,tmp,Temp,Dop=pertread(prefix,eigstep,eigint,eigpt,Ec,Ev,x,T,kb,eps0,q,h)
elif ratecode=='ELECTRA':
    tau,dop,tmp,Temp,Dop=elctread(elctcharge,eigstep,eigpt,Evwan,Eg,tdf1)
elif ratecode=='CUSTOM':
    tau,dop,tmp,Temp,Dop=customread(eigstep,eigpt,Ev,Evwan,Eg,tdf1)


mupt=math.floor((Eg+2*x)/mustep) 
mu=np.zeros(mupt)
#for i in range(0,mupt):
#    mu[i]=-x+mustep*i
mu=ChempotAr(mupt,x)
ndir=tdf2.shape[1]-1
#np.savetxt('tottdf.txt',TDFtot[:,:,0,0], delimiter=' ')
if ratecode!='CRTA':
    TDFtot=np.zeros((eigpt,10,dop,tmp))
    sig=np.zeros((mupt,10,dop,tmp)) #electrical conductivity
    Bmtr=np.zeros((mupt,10,dop,tmp))  #B matrix
    kmtr=np.zeros((mupt,10,dop,tmp))  #B matrix
    msig=np.zeros((3,3))
    mB=np.zeros((3,3))
    mk=np.zeros((3,3))
    mrho=np.zeros((mupt,dop,tmp,3,3))
    mS=np.zeros((mupt,dop,tmp,3,3))
    kappa=np.zeros((mupt,dop,tmp,3,3))
#Calling Total TDF
    TDFtot,Bext=TOTTDF(tdf1,tdf2,tau,Bex,nerbex,Evwan,dop,tmp)
#Calling DOS
    if bool(calcdos)==True:
        amsdos(amesh,eigstep,eigint,eigpt,T,kb,eps0,bni,bnf,Ev)

#Seebeck calculation
    if bool (calcseeb)==True:
        TDFseeb=np.zeros((eigpt,10,dop,tmp))
        sbsig=np.zeros((mupt,10,dop,tmp)) #electrical conductivity
        sbBmtr=np.zeros((mupt,10,dop,tmp))  #B matrix
        sbkmtr=np.zeros((mupt,10,dop,tmp))  #B matrix
        sbmsig=np.zeros((3,3))
        sbmB=np.zeros((3,3))
        sbmk=np.zeros((3,3))
        sbmrho=np.zeros((mupt,dop,tmp,3,3))
        sbmS=np.zeros((mupt,dop,tmp,3,3))
        kap=np.zeros((mupt,dop,tmp,3,3))
        TDFseeb=SEEBTDF(tdf1,tau,Evwan,dop,tmp)
        sbsig,sbBmtr,sbkmtr,sbmS,kap=intmatseeb(dop,tmp,mupt,ndir,eigpt,TDFseeb,mu,Temp,kb,eps0,sbsig,sbBmtr,sbkmtr,sbmsig,sbmB,sbmk,sbmrho,sbmS,kap)
        for idp in range(0,dop):
            for itmp in range(0,tmp):
                msfl="T{}dop{:.3e}S.txt".format(str(Temp[itmp]),Dop[idp,itmp])
                mrhfl="T{}dop{:.3e}Sig.txt".format(str(Temp[itmp]),Dop[idp,itmp])
                mefl="T{}dop{:.3e}Kap.txt".format(str(Temp[itmp]),Dop[idp,itmp])
                mwb="T{}dop{:.3e}Bmat.txt".format(str(Temp[itmp]),Dop[idp,itmp])
                mwk="T{}dop{:.3e}K.txt".format(str(Temp[itmp]),Dop[idp,itmp])
                mtff="T{}dop{:.3e}TDF1.txt".format(str(Temp[itmp]),Dop[idp,itmp])
                with open(msfl,'w') as sf:
                    sf.writelines('#Seebeck coefficient in the units of V/K ' + '\n')
                    sf.writelines('#Zero is top of the valence band ' + '\n')
                    sf.writelines('#mu(eV)	Sxx Sxy Sxz Syx Syy Syz Szx Szy Szz' + '\n')
                    np.savetxt(sf,np.c_[mu,sbmS[:,idp,itmp,0,0],sbmS[:,idp,itmp,0,1],sbmS[:,idp,itmp,0,2] ,sbmS[:,idp,itmp,1,0],sbmS[:,idp,itmp,1,1],sbmS[:,idp,itmp,1,2],sbmS[:,idp,itmp,2,0],sbmS[:,idp,itmp,2,1],sbmS[:,idp,itmp,2,2]], delimiter=' ')
                with open(mrhfl,'w') as rf:
                    rf.writelines('#Electrical conductivity in the units of 1/Ohm/m' + '\n')
                    rf.writelines('#Zero is top of the valence band ' + '\n')
                    rf.writelines('#mu(eV)	Sigxx Sigxy Sigxz Sigyx Sigyy Sigyz Sigzx Sigzy Sigzz' + '\n')
                    np.savetxt(rf,np.c_[mu,sbsig[:,1,idp,itmp],sbsig[:,2,idp,itmp],sbsig[:,3,idp,itmp],sbsig[:,4,idp,itmp],sbsig[:,5,idp,itmp],sbsig[:,6,idp,itmp],sbsig[:,7,idp,itmp],sbsig[:,8,idp,itmp],sbsig[:,9,idp,itmp]], delimiter=' ')
                with open(mefl,'w') as ef:
                    ef.writelines('#Electronic thermal conductivity in the units of W/m/K' + '\n')
                    ef.writelines('#Zero is top of the valence band ' + '\n')
                    ef.writelines('#mu(eV)	Kxx Kxy Kxz Kyx Kyy Kyz Kzx Kzy Kzz' + '\n')
                    np.savetxt(ef,np.c_[mu,kap[:,idp,itmp,0,0] ,kap[:,idp,itmp,0,1],kap[:,idp,itmp,0,2],kap[:,idp,itmp,1,0],kap[:,idp,itmp,1,1],kap[:,idp,itmp,1,2],kap[:,idp,itmp,2,0],kap[:,idp,itmp,2,1],kap[:,idp,itmp,2,2]], delimiter=' ')
                with open(mtff,'w') as tfs:
                    tfs.writelines('#Transport distribution function for the Seebeck effect in the units of C^2.S/m^3/kg' + '\n')
                    tfs.writelines('#Zero is the top of the valence band ' + '\n')
                    tfs.writelines('#E(eV)	TDFxx TDFxy TDFxz TDFyx TDFyy TDFyz TDFzx TDFzy TDFzz' + '\n')
                    np.savetxt(tfs,np.c_[TDFseeb[:,0,idp,itmp], TDFseeb[:,1,idp,itmp],TDFseeb[:,2,idp,itmp],TDFseeb[:,3,idp,itmp],TDFseeb[:,4,idp,itmp],TDFseeb[:,5,idp,itmp],TDFseeb[:,6,idp,itmp],TDFseeb[:,7,idp,itmp],TDFseeb[:,8,idp,itmp],TDFseeb[:,9,idp,itmp]], delimiter=' ')
                if bool(savematrix)==True:
                    with open(mwb,'w') as wb:
                        wb.writelines('#B matrix for the Seebeck effect in the units of V/K/Ohm/m' + '\n')
                        wb.writelines('#Zero is top of the valence band ' + '\n')
                        wb.writelines('#mu(eV)	Bxx Bxy Bxz Byx Byy Byz Bzx Bzy Bzz' + '\n')
                        np.savetxt(wb,np.c_[mu,sbBmtr[:,1,idp,itmp],sbBmtr[:,2,idp,itmp],sbBmtr[:,3,idp,itmp],sbBmtr[:,4,idp,itmp],sbBmtr[:,5,idp,itmp],sbBmtr[:,6,idp,itmp],sbBmtr[:,7,idp,itmp],sbBmtr[:,8,idp,itmp],sbBmtr[:,9,idp,itmp]], delimiter=' ')
                    with open(mwk,'w') as wk:
                        wk.writelines('#K matrix for the Seebeck effect in the units of W/m/K' + '\n')
                        wk.writelines('#Zero is top of the valence band ' + '\n')
                        wk.writelines('#mu(eV)	Kxx Kxy Kxz Kyx Kyy Kyz Kzx Kzy Kzz' + '\n')
                        np.savetxt(wk,np.c_[mu,sbkmtr[:,1,idp,itmp],sbkmtr[:,2,idp,itmp],sbkmtr[:,3,idp,itmp],sbkmtr[:,4,idp,itmp],sbkmtr[:,5,idp,itmp],sbkmtr[:,6,idp,itmp],sbkmtr[:,7,idp,itmp],sbkmtr[:,8,idp,itmp],sbkmtr[:,9,idp,itmp]], delimiter=' ')
 

#Calling integrating for sigma and B matrix
    sig,mrho,Bmtr,kmtr,mS,kappa=intmat(dop,tmp,mupt,ndir,eigpt,TDFtot,mu,Temp,kb,eps0,sig,Bmtr,kmtr,msig,mB,mk,mrho,mS,kappa)
    for idp in range(0,dop):
        for itmp in range(0,tmp):
            msfl="T{}dop{:.3e}NB{}.txt".format(str(Temp[itmp]),Dop[idp,itmp],Bex)
            mrhfl="T{}dop{:.3e}RB{}.txt".format(str(Temp[itmp]),Dop[idp,itmp],Bex)
            mefl="T{}dop{:.3e}EB{}.txt".format(str(Temp[itmp]),Dop[idp,itmp],Bex)
            mwb="T{}dop{:.3e}BmatB{}.txt".format(str(Temp[itmp]),Dop[idp,itmp],Bex)
            mwk="T{}dop{:.3e}KB{}.txt".format(str(Temp[itmp]),Dop[idp,itmp],Bex)
            mws="T{}dop{:.3e}SigB{}.txt".format(str(Temp[itmp]),Dop[idp,itmp],Bex)
            mtfs="T{}dop{:.3e}TDF2B{}.txt".format(str(Temp[itmp]),Dop[idp,itmp],Bex)
            with open(msfl,'w') as sf:
                sf.writelines('#Nernst coefficient ' + '\n')
                sf.writelines('#Zero is the top of the valence band ' + '\n')
                sf.writelines('#mu(eV)	N(V/K/T)' + '\n')
                np.savetxt(sf,np.c_[mu,mS[:,idp,itmp,1,0]/Bex], delimiter=' ')
            with open(mrhfl,'w') as rf:
                rf.writelines('#Hall coefficient' + '\n')
                rf.writelines('#Zero is the top of the valence band ' + '\n')
                rf.writelines('#mu(eV)	RH(Ohm.m/T)' + '\n')
                np.savetxt(rf,np.c_[mu,mrho[:,idp,itmp,1,0]/Bex], delimiter=' ')
            with open(mefl,'w') as ef:
                ef.writelines('#Ettingshausen coefficient' + '\n')
                ef.writelines('#Zero is the top of the valence band ' + '\n')
                ef.writelines('#mu(eV)	ETN(K.m/A/T)' + '\n')
                np.savetxt(ef,np.c_[mu,mS[:,idp,itmp,1,0]*Temp[itmp]/Bex/kappa[:,idp,itmp,1,1]], delimiter=' ')
            with open(mtfs,'w') as tfs:
                tfs.writelines('#Total transport distribution function for the Nernst effect in the units of C^2.S/m^3/kg' + '\n')
                tfs.writelines('#Zero is the top of the valence band ' + '\n')
                tfs.writelines('#E(eV)	TDFxx TDFxy TDFxz TDFyx TDFyy TDFyz TDFzx TDFzy TDFzz' + '\n')
                np.savetxt(tfs,np.c_[TDFtot[:,0,idp,itmp], TDFtot[:,1,idp,itmp],TDFtot[:,2,idp,itmp],TDFtot[:,3,idp,itmp],TDFtot[:,4,idp,itmp],TDFtot[:,5,idp,itmp],TDFtot[:,6,idp,itmp],TDFtot[:,7,idp,itmp],TDFtot[:,8,idp,itmp],TDFtot[:,9,idp,itmp]], delimiter=' ')
                if bool(savematrix)==True:
                    with open(mwb,'w') as wb:
                        wb.writelines('#B matrix for the Nernst effect V/K/Ohm/m' + '\n')
                        wb.writelines('#Zero is the top of the valence band' + '\n')
                        wb.writelines('#mu(eV)	Bxx Bxy Bxz Byx Byy Byz Bzx Bzy Bzz' + '\n')
                        np.savetxt(wb,np.c_[mu,Bmtr[:,1,idp,itmp],Bmtr[:,2,idp,itmp],Bmtr[:,3,idp,itmp],Bmtr[:,4,idp,itmp],Bmtr[:,5,idp,itmp],Bmtr[:,6,idp,itmp],Bmtr[:,7,idp,itmp],Bmtr[:,8,idp,itmp],Bmtr[:,9,idp,itmp]], delimiter=' ')
                    with open(mwk,'w') as wk:
                        wk.writelines('#K matrix for the Nernst effect in the units of W/m/K' + '\n')
                        wk.writelines('#Zero is the top of the valence band' + '\n')
                        wk.writelines('#mu(eV)	Kxx Kxy Kxz Kyx Kyy Kyz Kzx Kzy Kzz' + '\n')
                        np.savetxt(wk,np.c_[mu,kmtr[:,1,idp,itmp],kmtr[:,2,idp,itmp],kmtr[:,3,idp,itmp],kmtr[:,4,idp,itmp],kmtr[:,5,idp,itmp],kmtr[:,6,idp,itmp],kmtr[:,7,idp,itmp],kmtr[:,8,idp,itmp],kmtr[:,9,idp,itmp]], delimiter=' ')
                    with open(mws,'w') as ws:
                        ws.writelines('#Sigma matrix for the Nernst effect in the units of 1/Ohm/m' + '\n')
                        ws.writelines('#Zero is the top of the valence band' + '\n')
                        ws.writelines('#mu(eV)	Sigxx Sigxy Sigxz Sigyx Sigyy Sigyz Sigzx Sigzy Sigzz' + '\n')
                        np.savetxt(ws,np.c_[mu,sig[:,1,idp,itmp],sig[:,2,idp,itmp],sig[:,3,idp,itmp],sig[:,4,idp,itmp],sig[:,5,idp,itmp],sig[:,6,idp,itmp],sig[:,7,idp,itmp],sig[:,8,idp,itmp],sig[:,9,idp,itmp]], delimiter=' ')

else:
    tmp,Temp=TempAr(Tmin,Tmax,Tstep)
    TDFtot=np.zeros((eigpt,10))
    sig=np.zeros((mupt,10,tmp)) #electrical conductivity
    Bmtr=np.zeros((mupt,10,tmp))  #B matrix
    kmtr=np.zeros((mupt,10,tmp))  #B matrix
    msig=np.zeros((3,3))
    mB=np.zeros((3,3))
    mk=np.zeros((3,3))
    mrho=np.zeros((mupt,tmp,3,3))
    mS=np.zeros((mupt,tmp,3,3))
    kappa=np.zeros((mupt,tmp,3,3))
    TDFtot,Bext=CTTOTTDF(tdf1,tdf2,ctau,Bex,nerbex,Evwan)
    if bool(calcseeb)==True:
        TDFseeb=np.zeros((eigpt,10))
        sbsig=np.zeros((mupt,10,tmp)) #electrical conductivity
        sbBmtr=np.zeros((mupt,10,tmp))  #B matrix
        sbkmtr=np.zeros((mupt,10,tmp))  #B matrix
        sbmsig=np.zeros((3,3))
        sbmB=np.zeros((3,3))
        sbmk=np.zeros((3,3))
        sbmrho=np.zeros((mupt,tmp,3,3))
        sbmS=np.zeros((mupt,tmp,3,3))
        kap=np.zeros((mupt,tmp,3,3))
        TDFseeb=CTSEEBTDF(tdf1,Evwan,ctau)
        sbsig,sbBmtr,sbkmtr,sbmS,kap=CTintmatseeb(tmp,mupt,ndir,eigpt,TDFseeb,mu,Temp,kb,eps0,sbsig,sbBmtr,sbkmtr,sbmsig,sbmB,sbmk,sbmrho,sbmS,kap)
        for itmp in range(0,tmp):
            msfl="T{}S.txt".format(str(Temp[itmp]))
            mrhfl="T{}Sig.txt".format(str(Temp[itmp]))
            mefl="T{}Kap.txt".format(str(Temp[itmp]))
            mwb="T{}Bmat.txt".format(str(Temp[itmp]))
            mwk="T{}K.txt".format(str(Temp[itmp]))
            mtff="T{}TDF1.txt".format(str(Temp[itmp]))
            with open(msfl,'w') as sf:
                sf.writelines('#Seebeck coefficient in the units of V/K ' + '\n')
                sf.writelines('#Zero is top of the valence band ' + '\n')
                sf.writelines('#mu(eV)  Sxx Sxy Sxz Syx Syy Syz Szx Szy Szz' + '\n')
                np.savetxt(sf,np.c_[mu,sbmS[:,itmp,0,0],sbmS[:,itmp,0,1],sbmS[:,itmp,0,2] ,sbmS[:,itmp,1,0],sbmS[:,itmp,1,1],sbmS[:,itmp,1,2],sbmS[:,itmp,2,0],sbmS[:,itmp,2,1],sbmS[:,itmp,2,2]], delimiter=' ')
            with open(mrhfl,'w') as rf:
                rf.writelines('#Electrical conductivity in the units of 1/Ohm/m' + '\n')
                rf.writelines('#Zero is top of the valence band ' + '\n')
                rf.writelines('#mu(eV)  Sigxx Sigxy Sigxz Sigyx Sigyy Sigyz Sigzx Sigzy Sigzz' + '\n')
                np.savetxt(rf,np.c_[mu,sbsig[:,1,itmp],sbsig[:,2,itmp],sbsig[:,3,itmp],sbsig[:,4,itmp],sbsig[:,5,itmp],sbsig[:,6,itmp],sbsig[:,7,itmp],sbsig[:,8,itmp],sbsig[:,9,itmp]], delimiter=' ')
            with open(mefl,'w') as ef:
                ef.writelines('#Electronic thermal conductivity in the units of W/m/K' + '\n')
                ef.writelines('#Zero is top of the valence band ' + '\n')
                ef.writelines('#mu(eV)  Kxx Kxy Kxz Kyx Kyy Kyz Kzx Kzy Kzz' + '\n')
                np.savetxt(ef,np.c_[mu,kap[:,itmp,0,0] ,kap[:,itmp,0,1],kap[:,itmp,0,2],kap[:,itmp,1,0],kap[:,itmp,1,1],kap[:,itmp,1,2],kap[:,itmp,2,0],kap[:,itmp,2,1],kap[:,itmp,2,2]], delimiter=' ')
            with open(mtff,'w') as tfs:
                tfs.writelines('#Transport distribution function for the Seebeck effect in the units of C^2.S/m^3/kg' + '\n')
                tfs.writelines('#Zero is the top of the valence band ' + '\n')
                tfs.writelines('#E(eV)  TDFxx TDFxy TDFxz TDFyx TDFyy TDFyz TDFzx TDFzy TDFzz' + '\n')
                np.savetxt(tfs,np.c_[TDFseeb[:,0], TDFseeb[:,1],TDFseeb[:,2],TDFseeb[:,3],TDFseeb[:,4],TDFseeb[:,5],TDFseeb[:,6],TDFseeb[:,7],TDFseeb[:,8],TDFseeb[:,9]], delimiter=' ')
            if bool(savematrix)==True:
                with open(mwb,'w') as wb:
                    wb.writelines('#B matrix for the Seebeck effect in the units of V/K/Ohm/m' + '\n')
                    wb.writelines('#Zero is top of the valence band ' + '\n')
                    wb.writelines('#mu(eV)      Bxx Bxy Bxz Byx Byy Byz Bzx Bzy Bzz' + '\n')
                    np.savetxt(wb,np.c_[mu,sbBmtr[:,1,itmp],sbBmtr[:,2,itmp],sbBmtr[:,3,itmp],sbBmtr[:,4,itmp],sbBmtr[:,5,itmp],sbBmtr[:,6,itmp],sbBmtr[:,7,itmp],sbBmtr[:,8,itmp],sbBmtr[:,9,itmp]], delimiter=' ')
                with open(mwk,'w') as wk:
                    wk.writelines('#K matrix for the Seebeck effect in the units of W/m/K' + '\n')
                    wk.writelines('#Zero is top of the valence band ' + '\n')
                    wk.writelines('#mu(eV)      Kxx Kxy Kxz Kyx Kyy Kyz Kzx Kzy Kzz' + '\n')
                    np.savetxt(wk,np.c_[mu,sbkmtr[:,1,itmp],sbkmtr[:,2,itmp],sbkmtr[:,3,itmp],sbkmtr[:,4,itmp],sbkmtr[:,5,itmp],sbkmtr[:,6,itmp],sbkmtr[:,7,itmp],sbkmtr[:,8,itmp],sbkmtr[:,9,itmp]], delimiter=' ')
 #Calling integrating for sigma and B matrix
    sig,mrho,Bmtr,kmtr,mS,kappa=CTintmat(tmp,mupt,ndir,eigpt,TDFtot,mu,Temp,kb,eps0,sig,Bmtr,kmtr,msig,mB,mk,mrho,mS,kappa)
    for itmp in range(0,tmp):
        msfl="T{}NB{}.txt".format(str(Temp[itmp]),Bex)
        mrhfl="T{}RB{}.txt".format(str(Temp[itmp]),Bex)
        mefl="T{}EB{}.txt".format(str(Temp[itmp]),Bex)
        mwb="T{}BmatB{}.txt".format(str(Temp[itmp]),Bex)
        mwk="T{}KB{}.txt".format(str(Temp[itmp]),Bex)
        mws="T{}SigB{}.txt".format(str(Temp[itmp]),Bex)
        mtfs="T{}TDF2B{}.txt".format(str(Temp[itmp]),Bex)
        with open(msfl,'w') as sf:
            sf.writelines('#Nernst coefficient ' + '\n')
            sf.writelines('#Zero is the top of the valence band ' + '\n')
            sf.writelines('#mu(eV)      N(V/K/T)' + '\n')
            np.savetxt(sf,np.c_[mu,mS[:,itmp,1,0]/Bex], delimiter=' ')
        with open(mrhfl,'w') as rf:
            rf.writelines('#Hall coefficient' + '\n')
            rf.writelines('#Zero is the top of the valence band ' + '\n')
            rf.writelines('#mu(eV)      RH(Ohm.m/T)' + '\n')
            np.savetxt(rf,np.c_[mu,mrho[:,itmp,1,0]/Bex], delimiter=' ')
        with open(mefl,'w') as ef:
            ef.writelines('#Ettingshausen coefficient' + '\n')
            ef.writelines('#Zero is the top of the valence band ' + '\n')
            ef.writelines('#mu(eV)      ETN(K.m/A/T)' + '\n')
            np.savetxt(ef,np.c_[mu,mS[:,itmp,1,0]*Temp[itmp]/Bex/kappa[:,itmp,1,1]], delimiter=' ')
        with open(mtfs,'w') as tfs:
            tfs.writelines('#Total transport distribution function for the Nernst effect in the units of C^2.S/m^3/kg' + '\n')
            tfs.writelines('#Zero is the top of the valence band ' + '\n')
            tfs.writelines('#E(eV)      TDFxx TDFxy TDFxz TDFyx TDFyy TDFyz TDFzx TDFzy TDFzz' + '\n')
            np.savetxt(tfs,np.c_[TDFtot[:,0], TDFtot[:,1],TDFtot[:,2],TDFtot[:,3],TDFtot[:,4],TDFtot[:,5],TDFtot[:,6],TDFtot[:,7],TDFtot[:,8],TDFtot[:,9]], delimiter=' ')
            if bool(savematrix)==True:
                with open(mwb,'w') as wb:
                    wb.writelines('#B matrix for the Nernst effect V/K/Ohm/m' + '\n')
                    wb.writelines('#Zero is the top of the valence band' + '\n')
                    wb.writelines('#mu(eV)      Bxx Bxy Bxz Byx Byy Byz Bzx Bzy Bzz' + '\n')
                    np.savetxt(wb,np.c_[mu,Bmtr[:,1,itmp],Bmtr[:,2,itmp],Bmtr[:,3,itmp],Bmtr[:,4,itmp],Bmtr[:,5,itmp],Bmtr[:,6,itmp],Bmtr[:,7,itmp],Bmtr[:,8,itmp],Bmtr[:,9,itmp]], delimiter=' ')
                with open(mwk,'w') as wk:
                    wk.writelines('#K matrix for the Nernst effect in the units of W/m/K' + '\n')
                    wk.writelines('#Zero is the top of the valence band' + '\n')
                    wk.writelines('#mu(eV)      Kxx Kxy Kxz Kyx Kyy Kyz Kzx Kzy Kzz' + '\n')
                    np.savetxt(wk,np.c_[mu,kmtr[:,1,itmp],kmtr[:,2,itmp],kmtr[:,3,itmp],kmtr[:,4,itmp],kmtr[:,5,itmp],kmtr[:,6,itmp],kmtr[:,7,itmp],kmtr[:,8,itmp],kmtr[:,9,itmp]], delimiter=' ')
                with open(mws,'w') as ws:
                    ws.writelines('#Sigma matrix for the Nernst effect in the units of 1/Ohm/m' + '\n')
                    ws.writelines('#Zero is the top of the valence band' + '\n')
                    ws.writelines('#mu(eV)      Sigxx Sigxy Sigxz Sigyx Sigyy Sigyz Sigzx Sigzy Sigzz' + '\n')
                    np.savetxt(ws,np.c_[mu,sig[:,1,itmp],sig[:,2,itmp],sig[:,3,itmp],sig[:,4,itmp],sig[:,5,itmp],sig[:,6,itmp],sig[:,7,itmp],sig[:,8,itmp],sig[:,9,itmp]], delimiter=' ')

end = time.time()
print("The time of execution of above program is :",
      (end-start) , "s")
