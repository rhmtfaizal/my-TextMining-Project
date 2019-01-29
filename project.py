import math
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import csv
import re

def bacaData(doc):
    hasil=[]
    with open(doc,'r',newline='') as csvFile:
        baca = csv.reader(csvFile, delimiter=';')
        for row in baca:
            dataCSV=[]
            for i in range(len(row)):
                dataCSV.append(row[i])
            hasil.append(dataCSV)
    csvFile.close()
    return hasil

def bacaKolom(doc,n):
    hasil=[]
    for i in range(len(doc)):
        hasil.append(doc[i][n])
    return hasil

def bacakelas(doc):
    hasil = []
    for i in doc:
        token = re.findall('\d$',i)
        hasil.append(token)
    return hasil

def casefolding(doc):
    hasil = []
    for i in doc:
        d = i.lower()
        for j in range(len(d)):
            hasil_token = re.sub('[^a-z]',' ',d)
        hasil.append(hasil_token)
    return hasil

def tokenisasi(doc):
        hasil = []
        for i in range(len(doc)):
            doclower = doc[i].lower()
            hilangkarakter = re.sub('[^a-z0-9]',' ',doclower)
            split = hilangkarakter.split(' ')
            a = ''
            tokens = []
            for i in split:
                if i != a:
                    tokens.append(i)
            hasil.append(tokens)
        return hasil
    
def filtr(doc):
        filter = []
        with open('D:\KULIAH\Semester 7\TEXTMIN\project\stop_words.txt') as f:
            content = f.read()
        stoplist = content.split('\n')
        for i in range(len(doc)):
            temp = []
            for j in range(len(doc[i])):
                if doc[i][j] not in stoplist:
                    temp.append(doc[i][j])
            filter.append(temp)
        return filter

def Stemming(doc):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        hasilstem=[]
        for i in range(len(doc)):
            StemPerDoc=[]
            for j in range(len(doc[i])):
                temp = stemmer.stem(doc[i][j])
                StemPerDoc.append(temp)
            hasilstem.append(StemPerDoc)
        return hasilstem

def getTermUnik(doc):
        term = []
        for i in range(len(doc)):
            for j in range(len(doc[i])):
                if doc[i][j] not in term:
                    term.append(doc[i][j])
        return term
    
def countTF(corp, term):
    tf=[]
    for j in range(len(corp)):
        tfd={}
        for i in term:
        #menghitung TF jika ada
            tfs = corp[j].count(i)
            if tfs!=0:
                tfd[i]= 1+math.log10(tfs)
            else:
                tfd[i]= 0
        tf.append(tfd)
    return tf

def countDF(doc,term):
    hasil = {}
    for i in term:
        temp = 0
        for j in range(len(doc)):
            if doc[j][i] > 0:
                temp = temp + 1
                hasil[i] = temp
    return hasil

def getLengthData(doc):
    hasil = len(doc)
    return hasil

def countIDF(pjg,doc):
    hasil = {}
    for i in doc:
        hasil[i]= math.log10(pjg / float(doc[i]))
    return hasil
'''
def countTFIDF(tf,idf):
    hasil = {}
    for i in range(len(tf)):
        for j in tf[i]:
            hasil[i][j] = float(tf[i][j] * idf[j])
    return hasil
'''

def countTFIDF2(tf,idf):
    tfidf=[]
    for i in range(len(tf)):
        id={}
        for j in idf:
            id[j] = tf[i][j]*idf[j]
        tfidf.append(id)
    return tfidf

def normalisasi(tfidf):
    tfidfnew = tfidf
    w=0
    for i in range(len(tfidf)):
        for j in tfidf[i]:
            w = w+math.pow(tfidf[i][j],2)
        w = math.sqrt(w) #Penyebut
        for j in tfidf[i]:
            tfidfnew[i][j] = tfidf[i][j]/w
    return tfidfnew

def cossim(doclatih, docuji):
    dot = []
    for i in range(len(doclatih)):
        hasil=0
        for j in doclatih[i]:
            hasil+= doclatih[i][j] * docuji[j]
        dot.append(hasil)
    return dot

def sortingKNN(nilai,kelaslatih):
    hasil_sorting=[[]for k in range (len(nilai))]
    for i in range(len(nilai)):
        a = nilai[i]
        b = kelaslatih[i]
        hasil_sorting[i].append(a)
        hasil_sorting[i].append(b)
    hasil_sorting = sorted(hasil_sorting,reverse=True)
    return hasil_sorting

def getKlas(doc, K):
    hasil = []
    for i in range(len(doc)):
        temp = []
        for j in range(K):
            temp.append(doc[i][j][1])
        hasil.append(temp)
    return hasil

def voting(kelasuji,kelas):
    kelas_testing = {}
    hasilakhir =[]
    for i in kelas:
        kelas_testing[i]=0
    for i in range (len(kelasuji)):
        if kelasuji[i][0]in kelas:
            kelas_testing[kelasuji[i][0]] +=1
    kelasmax = max(kelas_testing['2'],kelas_testing['1'],kelas_testing['0'])
    for i in kelas_testing:
        if kelas_testing[i]==kelasmax:
            hasilakhir.append(i)
    print(kelas_testing)
    return hasilakhir[0]

def akurasi(hasilknn,kelas):
    hasil = 0
    for i in range(len(hasilknn)):
        if hasilknn[i] == kelas[i][0]:
            hasil += 1
    hasilakhir = (hasil/(len(hasilknn)))*100
    return hasilakhir

#Proses Training

doc = "F:/Downloads/dataset_sms_spam_bhs_indonesia/dataset_normalisasi_sms_spam _v11.csv"
data = bacaData(doc)
kolom = bacaKolom(data,0)
kelas = bacakelas(kolom)
case = casefolding(kolom)
token = tokenisasi(case)
file = filtr(token)
stem = Stemming(file)
termunik = getTermUnik(stem)
#print(termunik)
hasiltermfreq = countTF(stem,termunik)
hasildf = countDF(hasiltermfreq, termunik)
jumDok = getLengthData(stem)
hasilidf = countIDF(jumDok, hasildf)
hasiltfidf = countTFIDF2(hasiltermfreq, hasilidf)
hasilnormalisasi = normalisasi(hasiltfidf)
#print('hasil tfidf')
'''
for i in range(len(stem)):
    print("DOC",i+1,"\n",hasiltfidf[i],"\n")
    '''
for i in range(len(stem)):
    print("DOC",i+1,"\n",hasilnormalisasi[i],"\n")

#Proses Testings
doc1 = "F:/Downloads/dataset_sms_spam_bhs_indonesia/datauji_normalisasi.csv"
data1 = bacaData(doc1)
kolom1 = bacaKolom(data1,0)
kelas1 = bacakelas(kolom1)
case1 = casefolding(kolom1)
token1 = tokenisasi(case1)
file1 = filtr(token1)
stem1 = Stemming(file1)
hasiltermfreq1 = countTF(stem1, termunik)
hasiltfidf1 = countTFIDF2(hasiltermfreq1, hasilidf)
hasilnormalisasi1 = normalisasi(hasiltfidf1)
hasil = []
for i in range(len(hasilnormalisasi1)):
    hasilcossim= cossim(hasilnormalisasi, hasilnormalisasi1[i])
    hasil.append(hasilcossim)
    #print('Nilai Cossim Dokumen uji ',i+1,hasil[i],"\n")
#print(hasilcossim,"\n")
sort = []
for i in range(len(hasil)):
    hasilsorting = sortingKNN(hasil[i],kelas)
    sort.append(hasilsorting)
    #print('Dokumen uji ',i+1,sort[i],"\n")

kelas = ['2','1','0']
k = int(input('Masukkan nilai K: '))    
hasilkelas = getKlas(sort,k)
kelashasil = []
for i in range(len(hasilkelas)):
    coba = voting(hasilkelas[i],kelas)
    kelashasil.append(coba)
print('kelas prediksi: \n',kelashasil,'\n')
print('kelas asli: \n',kelas1)
nilaiakurasi = akurasi(kelashasil,kelas1)
print('Akurasi: ',nilaiakurasi,'%')
