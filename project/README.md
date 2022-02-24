# CS6910 Project

### [2022-01-28]
+ Questions to ask to TA:
  + What is the format of ASR's output ?
    + Is ASR a speech to text model ?
    + If we submit a hindi speech will ASR generate the output using roman dialect.
      + Depends on training data !!
  + Can we change our project if we are not able to find appropriate sentences ?
  + If same language can be selected by multiple teams ?
  + Whether speech processing should be done in the application or can be performed after data collection ?
    + We may have to do speech processing in the app, since the last date of feature adding is 15 Feb.

### [2022-01-26]
+ Domain:
  + Voice search: news, music
    + Social impact:
      + Such technology can be deployed in places where literacy rate in not sufficient. Using such tech, we may be able to provide them access to important news.
    + What is different from youtube audio search ?
      + Youtube audio search may not (verification needed ?) perform well on regional languages.
      + And we will only show verified news videos.
    + Sentences:
      + Tokens: music: general music keyword, news: politics, region specific
  + Complain filing: police complain, muncipality complain
    + Social impact:
      + People with low education level may not know the proper channel to communicate their problems. We can bidge this gap by enabling them in filing complains.
    + App:
      + Jan Sunyai, UP
    + Sentences:
      + Tokens: Crime, complain, police, municipality
      + Word dict: filter words from govt website.
  + Health Care:
    + for document purpose , medical history
    + Social impact:
      +  it create flexibility to provied efficient health care to rural areas.
    + Sentences
      + Token: health , disease, medical related terms.
+ Language:
  + Hindi or English (only one)

### [2022-01-21]
+ Mitesh Sir's Project link:
  + https://wandb.ai/miteshk/assignments/reports/CS6910-Project--VmlldzoxNDUyMzI0
+ Proposal [Jan 31]:
  + Use Latex, 1 page
+ TA : Tahir Javed
+ Task:
  + Language requiement:
    + Same language with multi team
    + Choose: Hindi, English
  + Choose your domain:
    + Eg: Digital Payment, hate speech detection, fake new
  + Based on domain choose sentenses:
    + https://indicnlp.ai4bharat.org/corpora/
  + TA will put the sentences to the app's server and then we need to generate the recordings
    + Do basic speech processing: to reduce noise
+ Schedule:
  + Once in 2 days.
