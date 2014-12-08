import urllib2
from bs4 import BeautifulSoup

f = open('pearljam.txt', 'w')

for y in range(1990, 2013):
        
        timestamp = str(y)
        print "Getting data for " + timestamp
        url = "http://www.setlist.fm/stats/pearl-jam-23d6b80b.html?year=" + str(y)
        page = urllib2.urlopen(url)
        soup = BeautifulSoup(page)
        
        songCnt_one=soup.findAll(attrs={"class":"songCount"})[1].span.string
        song_one=soup.findAll(attrs={"class":"songName"})[1].string
        songCnt_two=soup.findAll(attrs={"class":"songCount"})[3].span.string
        song_two=soup.findAll(attrs={"class":"songName"})[3].string
        songCnt_three=soup.findAll(attrs={"class":"songCount"})[5].span.string
        song_three=soup.findAll(attrs={"class":"songName"})[5].string
        songCnt_four=soup.findAll(attrs={"class":"songCount"})[7].span.string
        song_four=soup.findAll(attrs={"class":"songName"})[7].string
        songCnt_five=soup.findAll(attrs={"class":"songCount"})[9].span.string
        song_five=soup.findAll(attrs={"class":"songName"})[9].string
        songCnt_six=soup.findAll(attrs={"class":"songCount"})[11].span.string
        song_six=soup.findAll(attrs={"class":"songName"})[11].string
        songCnt_seven=soup.findAll(attrs={"class":"songCount"})[13].span.string
        song_seven=soup.findAll(attrs={"class":"songName"})[13].string
        songCnt_eight=soup.findAll(attrs={"class":"songCount"})[15].span.string
        song_eight=soup.findAll(attrs={"class":"songName"})[15].string
        songCnt_nine=soup.findAll(attrs={"class":"songCount"})[17].span.string
        song_nine=soup.findAll(attrs={"class":"songName"})[17].string

        
        yearstamp=str(y)
        timestamp=yearstamp

        f.write(timestamp + ',' + song_one + ',' + songCnt_one + ',' + song_two + ',' + songCnt_two
+ ',' + song_three + ',' + songCnt_three + ',' + song_four + ',' + songCnt_four + ',' + song_five + ',' + songCnt_five
+ ',' + song_six + ',' + songCnt_six + ',' + song_seven + ',' + songCnt_seven + ',' + song_eight + ',' + songCnt_eight
+ ',' + song_nine + ',' + songCnt_nine +'\n')

f.close()
