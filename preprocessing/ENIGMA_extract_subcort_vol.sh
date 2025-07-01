cd /media/lautaro/SSDLauti/reconallNCN 

echo "SubjID,LLatVent,RLatVent,Lthal,Rthal,Lcaud,Rcaud,Lput,Rput,Lpal,Rpal,Lhippo,Rhippo,Lamyg,Ramyg,Laccumb,Raccumb,ICV" > LandRvolumes.csv
for subj_id in `ls -d *`; do 
printf "%s,"  "${subj_id}" >> LandRvolumes.csv
for x in Left-Lateral-Ventricle Right-Lateral-Ventricle Left-Thalamus-Proper Right-Thalamus-Proper Left-Caudate Right-Caudate Left-Putamen Right-Putamen Left-Pallidum Right-Pallidum Left-Hippocampus Right-Hippocampus Left-Amygdala Right-Amygdala Left-Accumbens-area Right-Accumbens-area; do
printf "%g," `grep  ${x} ${subj_id}/stats/aseg.stats | awk '{print $4}'` >> LandRvolumes.csv
done
printf "%g" `cat ${subj_id}/stats/aseg.stats | grep IntraCranialVol | awk -F, '{print $4}'` >> LandRvolumes.csv
echo "" >> LandRvolumes.csv
done