function [Precision,Recall,TPR, FPR, AUC,AP,F] = QXL_ROC( image, hsegmap, NT )
%¼ÆËãÄ³Ò»·ùsmµÄROCÏà¹ØÊý¾Ý£¬µ«ÊÇÐèÒªAlgorithm_ROC.mÌá¹©ground truth¡£
%input parameter description: 
%image£ºÊäÈëµÄsm
%hsegmap£ºsm¶ÔÓ¦µÄÊÖ¶¯·Ö¸îÍ¼
%NT: ÓÐ¶àÉÙ¼¶»Ò¶ÈãÐÖµ
%output parameter description: 
%TPR, FPR£ºÕæÕæÂÊ£¬¼ÙÕæÂÊ£¬1*102
%AUC£ºROCÇúÏßÏÂ°üº¬µÄÃæ»ý£¬µ¥¸öÊýÖµ

%ÕâÀï¶ÔÊäÈëµÄsmºÍfixation map×öÁËÒ»¸öÍ³Ò»´¦Àí£¬±ä³ÉÈ«256¼¶»Ò¶ÈÍ¼Ïñ
img=mat2gray(image);

%img=uint8(img*(NT-1));


hsegmap=mat2gray(hsegmap);%ÕýÔò»¯µ½[0 1]
hsegmap=hsegmap(:,:,1);
img=mat2gray(imresize(img,size(hsegmap)));
img=(img*(NT-1));

positiveset  = hsegmap; %ÊÖ¶¯·Ö¸îÍ¼µÄÕæ¼¯ºÏ
negativeset = ~hsegmap ;%ÊÖ¶¯·Ö¸îÍ¼µÄ¼Ù¼¯ºÏ
P=sum(positiveset(:));%ÊÖ¶¯·Ö¸îÍ¼µÄÕæ¼¯ºÏµãµÄ¸öÊý
N=sum(negativeset(:));%ÊÖ¶¯·Ö¸îÍ¼µÄ¼Ù¼¯ºÏµãµÄ¸öÊý

%³õÊ¼»¯TPRºÍFPR£¬ÒòÎªÓÐ¶àÉÙ¸öãÐÖµËùÓÐ¾ÍÓÐ¶àÉÙ¶Ô[ TPR, FPR ]
TPR=zeros(1,NT);
FPR=zeros(1,NT);

Precision=zeros(1,NT);
F=zeros(1,NT);
%È·±£Ê×Î»ÊÇ1ºÍ0£¬Õâ¸ö²»Ó°ÏìµÃ·Ö£¬Ö»ÊÇÏÔÊ¾ÇúÏßÊ±ºÃ¿´
TPR(1)=1;
FPR(1)=1;
TPR(NT+2)=0;
FPR(NT+2)=0;
Precision(1)=0;
Precision(NT+2)=1;
Recall(1)=0;
Recall(NT+2)=1;


for i=1:NT+1
%¶ÔÊäÈëµÄsm£¬ÎÒÃÇ°ÑËü±ä³ÉÈ«100¼¶»Ò¶Èºó£¬¾ÍÄÃ0~99Õâ100¸ö»Ò¶È×÷ÎªãÐÖµ¡£¹ØÓÚãÐÖµµÄÑ¡È¡£¬ÎÒÔø¾­ÊÔ¹ýºÜ¶àÖÖ·½°¸£¬±ÈÈçºîÏþµÏPAMI2011
%µÄ´úÂë£¬Í³¼ÆsmÀïÓÐ¼¸¸ö²»ÖØ¸´µÄ»Ò¶È£¬ÄÃÕâÐ©»Ò¶È×öãÐÖµ£¬ÕâÑù×ö¼´¼´¾«È·ÓÖ½ÚÊ¡¼ÆËãÁ¿£¬µ«ÊÇ£¬ÓÉÓÚÃ¿·ùsmÀï²»ÖØ¸´µÄ»Ò¶È¸öÊý¸÷¸ö²»Í¬£¬¼Æ
%Ëã³öÀ´µÄTPRºÍFPRµÄ¸öÊýÒ²¾Í²»Í¬£¬½«À´Ã»·¨°Ñ¸÷¸ösmµÄTPR¡¢FPR×ö¾ùÖµºÍ·½²îÔËËã¡£»¹ÓÐÒ»ÖÖ·½·¨£¬½«smËùÓÐÏñËØµÄ»Ò¶È½øÐÐÅÅÐò£¬½«Õâ
%Ð©ÏñËØµãµÈ·Ö³ÉN·Ý£¬Ã¿·ÝÒ»¸öãÐÖµ£¬¹Ø¼üÊÇNÈ¡¶à´ó£¬NÈç¹ûÐ¡ÓÚsmµÄ²»ÖØ¸´µÄ»Ò¶È¸öÊý£¬ÄÇÃ´ROCÊýÖµºÍÇúÏß¾Í»áÆ«µÍ£¬µ«ÊÇÕâ¸ö¡°smµÄ²»ÖØ
%¸´µÄ»Ò¶È¸öÊý¡±Ã¿·ùÓÖ²»Í¬£¬Ö»ÄÜ°ÑNÈ¡´óÒ»µã£¬ºØÊ¤¾ÍÊÇÈ¡N=320£¬ÕâÃ÷ÏÔÊÇÀË·Ñ£¬ÒòÎªN>256µÄ»°¾ÍÃ»ÒâÒåÁË¡£ÎÒ¾­¹ýÊµÑé·¢ÏÖÈ¡100¸öãÐÖµ¼È
%¼õÉÙÔËËãÁ¿»¹¿ÉÒÔ±£³ÖÏàµ±µÄ¾«¶È
      T=i-1;
%³¬¹ýãÐÖµµÄ²¿·Ö¾ÍÊÇÖ÷¹ÛÈÏ¶¨µÄÕæ¼¯ºÏ
      positivesamples = img >= T;
%¼ÆËãÕæÕæºÍ¼ÙÕæ

      TPmat=positiveset.*positivesamples;
      FPmat=negativeset.*positivesamples;
      
       PS=sum(positivesamples(:));
       if PS~=0       
%Í³¼Æ¸÷ÏîÖ¸±êµÄ¾ßÌåÊýÖµ
      TP=sum(TPmat(:));
      FP=sum(FPmat(:));
%¼ÆËãÕæÕæÂÊºÍ¼ÙÕæÂÊ
      TPR(i+1)=TP/P;
      FPR(i+1)=FP/N;
      
      Precision(i+1)=TP/PS;
      Recall(i+1)=TPR(i+1);
%       F(i+1)=TP*Precision/(TP+0.3*Precision)*1.3;
      F(i+1)=TP*Precision/(TP+1*Precision)*2;
       end
end


%¼ÆËãAUC£¨ROCÇúÏßÏÂµÄÃæ»ý£©
AUC = -trapz(FPR, TPR);
AP = -trapz(TPR, Precision);

%F=mean(F(2:end-1));
end
