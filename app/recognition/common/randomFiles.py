
import glob
import random
path="../../alignment/Librispeech/train_clean_100/"
number_of_files=50
mylist = [f for f in glob.glob(path+"*/*/*.flac")]

files=[]
i=0
while i < number_of_files:
    r=random.randrange(0, len(mylist))
    if (mylist[r] not in files):
        files.append(mylist[r].replace("\\",'/'))
        i+=1
        print(files[-1])
    


result_path=[]
result_ref=[]
for wave_file in files:
    print(wave_file)
    result_path.append(wave_file+"\n")
    parts=wave_file.split("/")
    wave_file_name=parts[-1]
    wave_file_name_parts=wave_file_name.replace(".flac","").split("-")
    path_ref='/'.join(parts[:-1])+"/"+"-".join(wave_file_name_parts[:-1])+".trans.txt"
    f_ref=open(path_ref)
    for ref in f_ref.readlines():
        parts=ref.split(' ')
        name=parts[0]
        ref=" ".join(parts[1:])
        if ("-".join(wave_file_name_parts)==name):
            result_ref.append(ref)
            break

f=open("waves_test.txt","w")
f.writelines(result_path)
f.close()
f=open("ref.txt","w")
f.writelines(result_ref)
f.close()


