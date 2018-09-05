# Lexical-Ambiguity

How to use:

**python lexical.py**

Implementing the Vector Classification Model

I initialized all the "document" vectors, where each example sentence is its own document.
I also decided to use TF-IDF as default for all cases except for Naive Bayes classifier. I also stored all the lables indicated
in .I 1 1 into a array called sensenum. After that for the first 3600 training vectors I created two new vectors called
V_profile1 and V_profile2 with each being the average or centroid of all the trianing vectors according to their sense labels.
For the remaining 400 test vectors, used cosine similarity and also ordered the list by sim1-sim2 to visualize more easilly.
Large positive numbers indicate strongly sense1 and large negative numbers indicate strong sense2.
An example of the output is shown bellow.

Enter Option: 1
Choose file (plant/tank/perplace): plant
Choose stemmed or unstemmed: stemmed
   sense  predict  sim1-sim2   sim1     sim2    doc#    sense    title
   
================================================================================================================================
+    1      1       0.2136    0.2593   0.0458   3744  1  LIVING  of drying was observed in the PLANT grown in soil A than that
+    1      1       0.2127    0.2534   0.0407   3651  1  LIVING  tration at R6 , proportion of PLANT N from fixation , and seed
+    1      1       0.1966    0.2803   0.0836   3792  1  LIVING  ld per plant , internodes and PLANT height were reduced by pla
+    1      1       0.1894    0.2350   0.0456   3842  1  LIVING  reater number of branches per PLANT , more pods per plant , fe
+    1      1       0.1862    0.2344   0.0481   3771  1  LIVING  af , and number of leaves per PLANT ; however , P vulgaris pla
+    1      1       0.1828    0.2425   0.0597   3718  1  LIVING  ce the growth and health of a PLANT , including such inherited
+    1      1       0.1626    0.1836   0.0211   3745  1  LIVING                             As PLANT population increased , ear
+    1      1       0.1579    0.1858   0.0279   3948  1  LIVING  cted agronomic traits such as PLANT height , number of ear-bea
+    1      1       0.1525    0.1906   0.0381   3728  1  LIVING  , and the growth stage of the PLANT cells in the reaction mixt
+    1      1       0.1465    0.1656   0.0191   3712  1  LIVING  rotein , 1000 kernel weight , PLANT height , kernels head$\sp
+    1      1       0.1369    0.1681   0.0312   3924  1  LIVING  econditioning water stress on PLANT morphology , yield compone
+    1      1       0.1358    0.1635   0.0277   3834  1  LIVING  charge resulting in decreased PLANT available soil water </S>
+    1      1       0.1353    0.1629   0.0276   3757  1  LIVING  f grains in many parts of the PLANT , principally in embryonic
+    1      1       0.1295    0.1558   0.0263   3793  1  LIVING  eight in AS-D and lodging and PLANT appearance in AS-3 </S>
+    1      1       0.1273    0.1717   0.0444   3753  1  LIVING   function , soil physical and PLANT physiological aspects are
+    1      1       0.1247    0.1436   0.0189   3831  1  LIVING  hoot and leaflet size , total PLANT leaf area and in several r
+    1      1       0.1244    0.1417   0.0172   3710  1  LIVING  s , leaves , or fruits of the PLANT </S>
+    1      1       0.1234    0.1578   0.0344   3867  1  LIVING  acing , resulting in variable PLANT density </S>
+    1      1       0.1228    0.1784   0.0556   3904  1  LIVING  t improvement with increasing PLANT densities for three experi
+    1      1       0.1224    0.1492   0.0268   3789  1  LIVING  fa yield when S levels in the PLANT are below 2.5 mg S ( 0.25
+    1      1       0.1217    0.1401   0.0185   3828  1  LIVING  flowering the average sorghum PLANT had accumulated 63 % and 7
+    1      1       0.1216    0.1567   0.0350   3846  1  LIVING  tion achieved with the use of PLANT growth regulators </S>
+    1      1       0.1214    0.1592   0.0378   3732  1  LIVING  sect resistance , and general PLANT and line appearance </S>
+    1      1       0.1214    0.1607   0.0394   3681  1  LIVING   N ) , and the effects of the PLANT growth retardant paclobutr
+    1      1       0.1189    0.1341   0.0152   3883  1  LIVING  h H$\sp { + } $ released from PLANT roots </S>
+    1      1       0.1176    0.1532   0.0356   3707  1  LIVING  es that the cell or thallus ( PLANT body ) lacks : roots , ste
+    1      1       0.1146    0.1410   0.0264   3809  1  LIVING  buted to reduced Mn levels in PLANT tissue </S>
+    1      1       0.1125    0.1288   0.0162   3870  1  LIVING  species successfully promoted PLANT growth under highly fertil
+    1      1       0.1116    0.1902   0.0786   3759  1  LIVING  ment and the growth rate of a PLANT in isolation is a measure
+    1      1       0.1107    0.1358   0.0251   3912  1  LIVING  avity of seeds on emergence , PLANT characteristics , and yiel
+    1      1       0.1107    0.1275   0.0168   3953  1  LIVING  ) are essential nutrients for PLANT growth and development , p
+    1      1       0.1071    0.1291   0.0220   3769  1  LIVING   11.6 and 4.8 % reductions in PLANT height and 18.8 and 21.0 %
+    1      1       0.1051    0.1324   0.0273   3944  1  LIVING  er-soluble carbohydrate among PLANT parts at either 10 or 30C
+    1      1       0.1050    0.1375   0.0325   3612  1  LIVING  ( kg ) , dry weight ( % ) and PLANT volume ( m$\sp 3 $ ) </S>
+    1      1       0.1023    0.1048   0.0025   3954  1  LIVING  n had greater seed weight and PLANT height than Hobbit and Asg
+    1      1       0.1001    0.1279   0.0277   3914  1  LIVING  ium flux and shoot weight per PLANT indicated that selection f
+    1      1       0.0988    0.1109   0.0121   3756  1  LIVING  two N levels ( 40 or 200 mg N PLANT ( '-1 ) ) and four daytime
+    1      1       0.0985    0.1081   0.0096   3961  1  LIVING   number of seeds in NP21R and PLANT height and panicle length
+    1      1       0.0977    0.1429   0.0453   3847  1  LIVING   potential application to all PLANT species of such CLONING of
+    1      1       0.0973    0.1482   0.0509   3794  1  LIVING        Grafting is a method of PLANT propagation in which a sci
+    1      1       0.0972    0.1075   0.0103   3680  1  LIVING   In the low phosphorus soil , PLANT phosphorus uptake was incr
+    1      1       0.0962    0.1092   0.0129   3790  1  LIVING  he leaves to the grain during PLANT maturation </S>
+    1      1       0.0953    0.1260   0.0307   3767  1  LIVING  ee of damage depending on the PLANT species and the time at wh
+    1      1       0.0943    0.1053   0.0111   3754  1  LIVING  0$\sp5 $ g$\sp { -1 } $ fresh PLANT </S>
+    1      1       0.0940    0.1203   0.0263   3751  1  LIVING  different disease ratings and PLANT height was non significant
+    1      1       0.0935    0.1317   0.0382   3906  1  LIVING  morphological position on the PLANT was developed by which 65
+    1      1       0.0918    0.1177   0.0259   3807  1  LIVING   influence of the nematode on PLANT hose physiology might be p
+    1      1       0.0911    0.1327   0.0416   3952  1  LIVING   be continuous throughout the PLANT </S>
+    1      1       0.0911    0.1206   0.0296   3613  1  LIVING            The relationship of PLANT and weather measurements a
+    1      1       0.0909    0.1117   0.0208   3839  1  LIVING  ds contain two of the primary PLANT nutrients and are called m
+    1      1       0.0908    0.1348   0.0441   3758  1  LIVING  eatly stimulated the study of PLANT diseases Until 1900 , plan
+    1      1       0.0902    0.0909   0.0006   3605  1  LIVING  ipened ovary of any flowering PLANT , or ANGIOSPERM , and usua
+    1      1       0.0887    0.1283   0.0396   3634  1  LIVING  zation of nitrogen within the PLANT occurs during periods of l
+    1      1       0.0886    0.1533   0.0647   3797  1  LIVING  evels of various nutrients in PLANT parts </S>
+    1      1       0.0875    0.1221   0.0346   3926  1  LIVING  ntrolling growth of the whole PLANT </S>
+    1      1       0.0869    0.0928   0.0059   3825  1  LIVING  n of proteins that constitute PLANT and animal cells </S>
+    1      1       0.0850    0.1238   0.0387   3845  1  LIVING  ween soil taxonomic units and PLANT associations ; and discrim
+    1      1       0.0850    0.1469   0.0618   3925  1  LIVING  edures that constitute modern PLANT breeding </S>
+    1      1       0.0850    0.1182   0.0333   3875  1  LIVING  rry ( Vaccinium ashei Reade ) PLANT survival , growth , develo
+    1      1       0.0836    0.1054   0.0218   3673  1  LIVING  d to merge cells of different PLANT species </S>
+    1      1       0.0830    0.1049   0.0219   3826  1  LIVING  in BS10 and BS11 mean ear and PLANT height decreased from the
+    1      1       0.0825    0.0931   0.0106   3721  1  LIVING  o 0.75 m from the stem of the PLANT </S>
+    1      1       0.0825    0.0863   0.0037   3642  1  LIVING  ed cheat spike number , cheat PLANT dry weight , and cheat see
+    1      1       0.0818    0.1173   0.0355   3832  1  LIVING  cycling of minerals by eating PLANT parts and then excreting s
+    1      1       0.0813    0.1053   0.0240   3746  1  LIVING  on data , in conjunction with PLANT height data , allowed dete
+    1      1       0.0792    0.1099   0.0307   3879  1  LIVING  mine water stress in numerous PLANT species </S>
+    1      1       0.0790    0.0927   0.0137   3830  1  LIVING  ssortment of animals and some PLANT material and are found nea
+    1      1       0.0782    0.1045   0.0264   3686  1  LIVING         Research revealed that PLANT extracts had little hypogl
+    1      1       0.0782    0.1081   0.0300   3682  1  LIVING  ve photosynthetic area of the PLANT </S>
+    1      1       0.0778    0.1138   0.0360   3708  1  LIVING  t experiment evaluated single PLANT response to treatment and
+    1      1       0.0741    0.1057   0.0316   3763  1  LIVING   1 ) determine the effects of PLANT cover and trampling by liv
+    1      1       0.0733    0.1049   0.0316   3755  1  LIVING  ithin the tissues of the host PLANT </S>
+    1      1       0.0725    0.1100   0.0375   3640  1  LIVING  tems used only two kingdoms , PLANT and animal , whereas most
+    1      1       0.0719    0.0822   0.0102   3852  1  LIVING  tly associated with decreased PLANT vigor </S>
+    1      1       0.0718    0.0743   0.0025   3717  1  LIVING  odule number , and nodule and PLANT dry weight </S>
+    1      1       0.0717    0.0944   0.0227   3674  1  LIVING               All parts of the PLANT , particularly the berries
+    1      1       0.0713    0.1177   0.0463   3713  1  LIVING   years were lower in soil and PLANT Zn than other fields </S>
+    1      1       0.0707    0.0869   0.0162   3959  1  LIVING  itivity to APAR under varying PLANT and soil moisture conditio
+    1      1       0.0693    0.1063   0.0370   3876  1  LIVING  opulation with a greater mean PLANT size and reduced coefficie
+    1      1       0.0676    0.1026   0.0350   3750  1  LIVING  ount of leaf area , and total PLANT weight followed by sicklep
+    1      1       0.0665    0.1067   0.0402   3977  1  LIVING   they have been troubled with PLANT diseases </S>
+    1      1       0.0664    0.0972   0.0308   3960  1  LIVING  aters Sulfur occurs in living PLANT and animal tissue as part
+    1      1       0.0644    0.1118   0.0474   3684  1  LIVING  fruit decay , reduced size of PLANT , matured sooner and deter
+    1      1       0.0642    0.1130   0.0488   3958  1  LIVING  -stem portion of the 1-yr-old PLANT and favored quinine produc
+    1      1       0.0641    0.0936   0.0295   3726  1  LIVING  those substances derived from PLANT and animal ( organic ) sou
+    1      1       0.0623    0.0935   0.0312   3675  1  LIVING  the riparian and non-riparian PLANT communities </S>
+    1      1       0.0619    0.0664   0.0045   3915  1  LIVING   : pollen is carried from one PLANT to another by wind or inse
+    1      1       0.0618    0.0948   0.0330   3645  1  LIVING   important predictor of final PLANT uplift , was positively co
+    1      1       0.0613    0.0924   0.0311   3752  1  LIVING                  In most other PLANT groups the leaves enlarge
+    1      1       0.0610    0.0707   0.0096   3637  1  LIVING  s for resistance to tarnished PLANT bug </S>
+    1      1       0.0609    0.0795   0.0187   3947  1  LIVING  ned included effects of : the PLANT hormone auxin ; buffers of
+    1      1       0.0608    0.0943   0.0335   3918  1  LIVING  evelopment and differences in PLANT size </S>
+    1      1       0.0596    0.0666   0.0070   3731  1  LIVING                 Three distinct PLANT viruses , transmitted by t
+    1      1       0.0586    0.0853   0.0267   3647  1  LIVING  itutes substantive proof that PLANT breeding was practiced by
+    1      1       0.0580    0.0781   0.0201   3872  1  LIVING  reduced small yellow nutsedge PLANT size by more than 50 % </S
+    1      1       0.0573    0.0599   0.0027   3602  1  LIVING  dine and deoxycytidine rescue PLANT growth and differentiation
+    1      1       0.0560    0.0658   0.0098   3922  1  LIVING                     The cotton PLANT grows upright to a height
+    1      1       0.0557    0.0887   0.0330   3873  1  LIVING  re low in nutrients The onion PLANT is potentially a biennial
+    1      1       0.0552    0.0766   0.0215   3762  1  LIVING  ction of nodABC expression by PLANT flavonoids , and probably
+    1      1       0.0548    0.0644   0.0096   3679  1  LIVING   did not reverse to untreated PLANT levels with an IE and was
+    1      1       0.0529    0.0646   0.0117   3878  1  LIVING  essive epiphytic colonizer of PLANT tissues </S>
+    1      1       0.0524    0.0828   0.0304   3720  1  LIVING  gronomic characters including PLANT height and days to 50 % si
+    1      1       0.0523    0.0759   0.0235   3611  1  LIVING  uence of the inability of the PLANT to extract 100 % of the wa
+    1      1       0.0515    0.0566   0.0051   3764  1  LIVING  trong correlation between the PLANT characteristics that were
+    1      1       0.0513    0.0706   0.0193   3833  1  LIVING          Heritability of these PLANT parameters which could be
+    1      1       0.0501    0.0580   0.0078   3950  1  LIVING  spatial variability ( HH ) in PLANT species and forest structu
+    1      1       0.0491    0.0967   0.0476   3709  1  LIVING  lk position were utilized for PLANT samples </S>
+    1      1       0.0474    0.0933   0.0459   3986  1  LIVING   human deaths attributable to PLANT poisons is insignificant <
+    1      1       0.0455    0.0759   0.0304   3910  1  LIVING  In 1979 , pitted morningglory PLANT weights were reduced by 66
+    1      1       0.0452    0.0817   0.0366   3913  1  LIVING  ural carbohydrate ( NSC ) per PLANT but did influence NSC part
+    1      1       0.0449    0.1184   0.0736   3978  1  LIVING  shed as vital repositories of PLANT seeds ; a large genetic po
+    1      1       0.0447    0.0644   0.0197   3946  1  LIVING                      Efficient PLANT regeneration was obtained
+    1      1       0.0435    0.0954   0.0519   3993  1  LIVING  s or eventually recycled into PLANT matter </S>
+    1      1       0.0430    0.0661   0.0231   3688  1  LIVING                            The PLANT compartment is considered
+    1      1       0.0425    0.0689   0.0263   3919  1  LIVING  und , coverages of individual PLANT species , and screening co
+    1      1       0.0418    0.0579   0.0161   3917  1  LIVING            Results showed that PLANT roots contain AcP , NP , a
+    1      1       0.0414    0.0654   0.0241   3685  1  LIVING  atpea changed with respect to PLANT organs and experimental fa
+    1      1       0.0412    0.0740   0.0328   3880  1  LIVING  cy , and importance values of PLANT species , including Trautv
+    1      1       0.0411    0.0691   0.0280   3646  1  LIVING   , sea spray , bacteria , and PLANT spores </S>
+    1      1       0.0410    0.0605   0.0195   3851  1  LIVING  presence of natural tarnished PLANT bug infestations </S>
+    1      1       0.0407    0.0998   0.0591   3972  1  LIVING   techniques for manufacturing PLANT protein products are large
+    1      1       0.0405    0.0853   0.0449   3716  1  LIVING  corn stand using the isolated PLANT as a model </S>
+    1      1       0.0402    0.0508   0.0106   3808  1  LIVING  ating a crucial role for this PLANT organ in the perception of
*    2      1       0.0401    0.1147   0.0745   3996  2  FACTORY                     The total PLANT population consisted of 24
+    1      1       0.0397    0.0752   0.0355   3715  1  LIVING  th traits in order to aid the PLANT breeder </S>
+    1      1       0.0392    0.0863   0.0471   3691  1  LIVING            The general goal of PLANT breeding is to assemble in
+    1      1       0.0390    0.0483   0.0093   3607  1  LIVING   and K8 had fewer nodules per PLANT than Cunningham and K67 bu
+    1      1       0.0386    0.0484   0.0098   3770  1  LIVING  ph Botrytis fabae on fhe host PLANT Vicia faba were studied </
*    2      1       0.0382    0.0584   0.0202   3969  2  FACTORY                   A large MSF PLANT may have 60 flash chambers
+    1      1       0.0378    0.0596   0.0219   3711  1  LIVING  ound to be poor predictors of PLANT preferences except in tria
+    1      1       0.0373    0.0552   0.0179   3603  1  LIVING  th 300 ppm N Foliar color and PLANT quality was highest with 3
+    1      1       0.0372    0.0478   0.0106   3804  1  LIVING                     The entire PLANT is usually pure white , or
+    1      1       0.0370    0.0604   0.0234   3766  1  LIVING  s of the C ( ,3 ) terrestrial PLANT , Asparagus sprengeri Rege
+    1      1       0.0358    0.0593   0.0235   3725  1  LIVING  y be used to repair a type of PLANT damage known as girdling ,
+    1      1       0.0357    0.0449   0.0092   3609  1  LIVING   bottoms for organic ( mostly PLANT ) material , which they st
+    1      1       0.0352    0.0597   0.0245   3733  1  LIVING  water from a nylon processing PLANT was applied to 'Ky 31 ' ta
+    1      1       0.0350    0.0630   0.0279   3868  1  LIVING  ormonal control of growth The PLANT body of a hornwort , or ga
+    1      1       0.0349    0.0596   0.0247   3722  1  LIVING   production with no effect on PLANT vigor </S>
+    1      1       0.0346    0.1145   0.0800   3641  1  LIVING  an cell wall thickness of the PLANT tissues than to any of the
+    1      1       0.0334    0.0530   0.0196   3760  1  LIVING  nts derived from the FOXGLOVE PLANT , including digitoxin , th
+    1      1       0.0322    0.0471   0.0149   3965  1  LIVING  M ) was used to analyze pilot PLANT experimental data </S>
+    1      1       0.0318    0.0410   0.0091   3601  1  LIVING  eaflets sampled from the same PLANT </S>
+    1      1       0.0312    0.0808   0.0496   3987  1  LIVING  er " as an integral part of a PLANT </S>
+    1      1       0.0310    0.0395   0.0084   3802  1  LIVING  and the habitat of the parent PLANT can often be inferred from
+    1      1       0.0310    0.0388   0.0078   3905  1  LIVING  ad penetrated walls of living PLANT cells and ramified intra -
+    1      1       0.0309    0.0522   0.0212   3644  1  LIVING  orted for vertebrate and some PLANT mitochondria , and lack of
+    1      1       0.0308    0.1017   0.0708   3848  1  LIVING  e quantities of wood or other PLANT materials ( for a time , k
+    1      1       0.0308    0.0763   0.0455   3942  1  LIVING  ation has been limited by the PLANT 's vulnerability to virus
+    1      1       0.0308    0.0739   0.0431   3943  1  LIVING  g to avoid being eaten , many PLANT species have developed som
*    2      1       0.0308    0.0528   0.0220   3988  2  FACTORY                           The PLANT used a lacquer coating on
+    1      1       0.0293    0.0512   0.0220   3689  1  LIVING   Rieske iron-sulfur center in PLANT mitochondria , and a diffe
+    1      1       0.0292    0.0300   0.0008   3799  1  LIVING  us esculentus , is a tropical PLANT grown as a vegetable </S>
+    1      1       0.0290    0.0923   0.0632   3824  1  LIVING  e short term had no effect on PLANT survival , two consecutive
+    1      1       0.0289    0.0412   0.0123   3761  1  LIVING  osphorus concentration in all PLANT parts </S>
+    1      1       0.0285    0.0310   0.0024   3639  1  LIVING                  Derris is any PLANT of the genus Derris in the
+    1      1       0.0275    0.0770   0.0495   3843  1  LIVING  d when 1 of the stems on each PLANT were clipped at each cut <
+    1      1       0.0273    0.0617   0.0344   3606  1  LIVING  pening phase , a mechanism of PLANT bug tolerance </S>
+    1      1       0.0261    0.0473   0.0212   3677  1  LIVING   , C canadensis--a herbaceous PLANT , with woody rootstock , w
+    1      1       0.0258    0.0280   0.0022   3719  1  LIVING            Nitrogen loading as PLANT detritus into hyacinth pon
+    1      1       0.0244    0.0546   0.0302   3795  1  LIVING   show relationships among the PLANT communities and to identif
+    1      1       0.0242    0.1149   0.0907   3990  1  LIVING  most unlimited potential that PLANT breeding provides , and th
+    1      1       0.0241    0.0460   0.0219   3968  1  LIVING  ceptions of the importance of PLANT science competencies from
+    1      1       0.0239    0.0397   0.0158   3916  1  LIVING  ent in diploid and tetraploid PLANT sections were supportive o
+    1      1       0.0238    0.0452   0.0214   3827  1  LIVING                      Among the PLANT eaters were a type of hors
+    1      1       0.0234    0.0540   0.0306   3678  1  LIVING  ed more than 60 % of the same PLANT communities each season ,
+    1      1       0.0232    0.0426   0.0194   3608  1  LIVING  ometimes known as the tapioca PLANT , is a member of the spurg
+    1      1       0.0231    0.0354   0.0123   3800  1  LIVING  ber of the potyvirus group of PLANT viruses </S>
+    1      1       0.0231    0.0374   0.0144   3837  1  LIVING  the bast fibers from the flax PLANT </S>
+    1      1       0.0224    0.0684   0.0460   3772  1  LIVING  ontaining natural products of PLANT origin that have an alkali
+    1      1       0.0217    0.0392   0.0175   3803  1  LIVING  st-size leaf particle The tea PLANT found in Taiwan and south
+    1      1       0.0215    0.0395   0.0180   3635  1  LIVING  igands to immobilize isolated PLANT RNA polymerase II and asso
+    1      1       0.0209    0.0515   0.0306   3869  1  LIVING  onditions as similar existing PLANT assemblages do today </S>
+    1      1       0.0207    0.0213   0.0006   3654  1  LIVING  uelen cabbage is a herbaceous PLANT , Pringlea antiscorbutica
+    1      1       0.0207    0.0334   0.0128   3908  1  LIVING  or Theresa Arnold , the agave PLANT has special significance a
+    1      1       0.0202    0.0365   0.0164   3963  1  LIVING   , however , cross reacted to PLANT antigens and thus are unsu
+    1      1       0.0201    0.0293   0.0092   3727  1  LIVING  is a tall , tropical , annual PLANT in the caper family , Capp
+    1      1       0.0196    0.0530   0.0334   3773  1  LIVING  mpounds are a large family of PLANT chemicals composed of a su
*    2      1       0.0192    0.0310   0.0118   3747  2  FACTORY          Change occurs at the PLANT ( speculation ) , or at th
+    1      1       0.0187    0.0761   0.0574   3692  1  LIVING  vantages in converting scarce PLANT resources into food </S>
+    1      1       0.0186    0.0297   0.0111   3841  1  LIVING  gus or pickled , or the whole PLANT ground and made into mush
+    1      1       0.0185    0.0353   0.0169   3994  1  LIVING  The prey trips a lever on the PLANT " door , " which allows wa
*    2      1       0.0182    0.0783   0.0600   3798  2  FACTORY on 's influence on industrial PLANT location is currently unde
+    1      1       0.0174    0.0294   0.0120   3874  1  LIVING  s , and many other xerophytic PLANT foods </S>
+    1      1       0.0168    0.0396   0.0228   3945  1  LIVING  ion experiments , making this PLANT a potential virus reservoi
+    1      1       0.0166    0.0661   0.0495   3643  1  LIVING   was transferred to a tobacco PLANT </S>
*    2      1       0.0157    0.0638   0.0481   3970  2  FACTORY ome supplies from the Spanish PLANT , an acid-like substance n
+    1      1       0.0148    0.0293   0.0145   3723  1  LIVING  n various parts of the potato PLANT was quantitated by rocket
+    1      1       0.0147    0.0412   0.0265   3838  1  LIVING  antisera raised against whole PLANT protein antigens </S>
+    1      1       0.0146    0.0343   0.0197   3956  1  LIVING                     Ranking of PLANT associates and environment
+    1      1       0.0143    0.0253   0.0110   3871  1  LIVING   is one of the weakest of the PLANT fibers , its use is limite
+    1      1       0.0142    0.0461   0.0320   3676  1  LIVING  age utilization or for future PLANT improvement </S>
+    1      1       0.0137    0.0236   0.0099   3911  1  LIVING                      In higher PLANT membranes , lecithin and P
+    1      1       0.0121    0.0240   0.0119   3687  1  LIVING                            The PLANT 's essential oil is used i
*    2      1       0.0118    0.0411   0.0293   3630  2  FACTORY nd told us a sewage-treatment PLANT was going in " </S>
+    1      1       0.0117    0.0610   0.0493   3823  1  LIVING  yford scientists identified a PLANT gene that produces " true
+    1      1       0.0112    0.0611   0.0499   3951  1  LIVING         In extreme cases the " PLANT " may actually develop int
*    2      1       0.0112    0.0519   0.0408   3979  2  FACTORY oes on : " But our Fort Myers PLANT , because of the warm wate
+    1      1       0.0110    0.0762   0.0651   3982  1  LIVING  ers that collect and preserve PLANT germ plasm </S>
+    1      1       0.0109    0.0498   0.0389   3749  1  LIVING  d more than merely supplement PLANT foods with animal products
+    1      1       0.0105    0.0260   0.0155   3981  1  LIVING  loration Hyssop is an ancient PLANT and has been considered sa
+    1      1       0.0100    0.0392   0.0292   3881  1  LIVING  rly mildew , Oidium , and the PLANT louse , Phylloxera </S>
+    1      1       0.0098    0.0972   0.0874   3976  1  LIVING  of the seed of new , improved PLANT varieties </S>
+    1      1       0.0094    0.0196   0.0101   3610  1  LIVING   in mixed borders or as a pot PLANT 0277750-0 Stonehenge </S>
+    1      1       0.0090    0.0201   0.0111   3849  1  LIVING                           This PLANT is native to Colorado and
+    1      1       0.0083    0.0279   0.0196   3796  1  LIVING  carved or etched naturalistic PLANT and animal forms </S>
*    2      1       0.0076    0.0943   0.0867   3724  2  FACTORY e one Quality of Working Life PLANT </S>
*    2      1       0.0076    0.0998   0.0922   3683  2  FACTORY  the grain volume than single PLANT firms </S>
*    2      1       0.0069    0.0618   0.0549   3696  2  FACTORY al director , said the Newark PLANT had frequently been used t
*    2      1       0.0068    0.0439   0.0370   3805  2  FACTORY ls in bleaching and the pilot PLANT trial in thermomechanical
+    1      1       0.0058    0.0416   0.0358   3765  1  LIVING  viously ungrazed portion of a PLANT than the part which had be
*    2      1       0.0057    0.0159   0.0102   3955  2  FACTORY icinity of a titanium dioxide PLANT in the Gulf of Bothnia </S
+    1      1       0.0045    0.0356   0.0310   3975  1  LIVING  basis for his hypothesis that PLANT domestication and early fa
*    2      1       0.0044    0.0413   0.0370   3736  2  FACTORY        " Several parts of the PLANT are fairly hazardous , inc
*    2      1       0.0042    0.0220   0.0178   3714  2  FACTORY lso investigated explosives , PLANT chemistry , and the histor
+    1      1       0.0042    0.0496   0.0454   3957  1  LIVING  se it was once extracted from PLANT ashes ; now almost all is
*    2      1       0.0036    0.0388   0.0352   3973  2  FACTORY  of such maneuvers , the main PLANT has experienced a parts sh
*    2      1       0.0032    0.0197   0.0165   3812  2  FACTORY n fish sticks in a processing PLANT , they can't sound like st
+    1      1       0.0015    0.0691   0.0676   3995  1  LIVING  assess the impact of physical PLANT deferred maintenance , and
+    1      1       0.0001    0.0408   0.0407   3850  1  LIVING  ncountered by geneticists and PLANT breeders working with cott
*    1      2      -0.0002    0.0226   0.0228   3748  1  LIVING  corated with brightly painted PLANT motifs , especially papyru
+    2      2      -0.0008    0.0162   0.0170   3853  2  FACTORY  addition of a Ralston Purina PLANT , it became the world 's b
*    1      2      -0.0010    0.0460   0.0470   3984  1  LIVING  mpotence , and so long as the PLANT remained rare in Europe ,
+    2      2      -0.0026    0.0241   0.0267   3652  2  FACTORY tant limits are viewed by the PLANT as prima facie evidence of
*    1      2      -0.0032    0.0439   0.0471   3840  1  LIVING                            The PLANT remains in bloom for about
*    1      2      -0.0060    0.0355   0.0416   3966  1  LIVING  al release , is maintained in PLANT breeding </S>
*    1      2      -0.0061    0.0393   0.0454   3974  1  LIVING               Besides offering PLANT displays , many have botan
+    2      2      -0.0063    0.0248   0.0311   3648  2  FACTORY e Oak Ridge gaseous diffusion PLANT </S>
*    1      2      -0.0063    0.0773   0.0836   3992  1  LIVING  h facilities of importance in PLANT taxonomy </S>
+    2      2      -0.0076    0.0271   0.0347   3866  2  FACTORY  , established a meat-packing PLANT to supply eastern markets
+    2      2      -0.0082    0.0296   0.0378   3971  2  FACTORY           And like the Soviet PLANT , it lacks a thick , concr
*    1      2      -0.0084    0.0859   0.0943   3983  1  LIVING  se have been developed in the PLANT as a defense against preda
*    1      2      -0.0085    0.0293   0.0378   3980  1  LIVING  boldt laid the foundations of PLANT geography , performing fie
+    2      2      -0.0088    0.0055   0.0143   3858  2  FACTORY heir delivery to the Winnipeg PLANT would be delayed because o
+    2      2      -0.0091    0.0662   0.0753   3661  2  FACTORY oup of 67 women at the Hudson PLANT working in so-called clean
*    1      2      -0.0103    0.0311   0.0414   3694  1  LIVING   used it to make a " pomato " PLANT , although its tomatoes an
+    2      2      -0.0107    0.1012   0.1119   3877  2  FACTORY ngle product may occur at the PLANT in each time period </S>
+    2      2      -0.0109    0.0011   0.0120   3738  2  FACTORY Metros at Rover 's Longbridge PLANT near Birmingham , is one o
+    2      2      -0.0111    0.0450   0.0560   3629  2  FACTORY ones in a vat of steel in the PLANT 's blastfurnace area </S>
+    2      2      -0.0124    0.0238   0.0361   3649  2  FACTORY s who , in turn , shipped the PLANT to Europe as a high-priced
+    2      2      -0.0146    0.0311   0.0456   3650  2  FACTORY n enrollment and the physical PLANT while financial problems w
*    1      2      -0.0147    0.0182   0.0328   3614  1  LIVING                   The story of PLANT breeding since 1920 or so
+    2      2      -0.0151    0.0471   0.0622   3962  2  FACTORY nt light tones--such as the " PLANT light ' that mimics sunlig
+    2      2      -0.0155    0.0527   0.0683   3844  2  FACTORY zation design of the chemical PLANT and traditional union stra
+    2      2      -0.0157    0.0424   0.0580   3604  2  FACTORY                      For most PLANT tissues , construction cos
*    1      2      -0.0167    0.0577   0.0745   3964  1  LIVING  te the carbon influx into the PLANT </S>
+    2      2      -0.0184    0.0623   0.0807   3967  2  FACTORY rm products and soil near the PLANT have almost returned to no
+    2      2      -0.0195    0.0499   0.0694   3989  2  FACTORY                           The PLANT will produce parts for the
+    2      2      -0.0208    0.0080   0.0288   3817  2  FACTORY of improper procedures at the PLANT and allowed them to contin
+    2      2      -0.0229    0.0166   0.0395   3655  2  FACTORY  , 1981 , nine days after the PLANT 's license was revoked </S
+    2      2      -0.0230    0.0181   0.0411   3657  2  FACTORY rchild 's Long Island , N.Y , PLANT , emanates from New York l
+    2      2      -0.0231    0.0031   0.0261   3667  2  FACTORY t a coroner 's inquest at the PLANT this morning </S>
+    2      2      -0.0242    0.0444   0.0686   3829  2  FACTORY ments made by supervisors and PLANT managers ; and paternalism
+    2      2      -0.0252    0.0284   0.0535   3949  2  FACTORY ts and management of physical PLANT are presented </S>
+    2      2      -0.0253    0.0400   0.0653   3801  2  FACTORY  margin , debt ratio , age of PLANT , and current ratio </S>
+    2      2      -0.0258    0.0506   0.0764   3730  2  FACTORY hm using data measured at the PLANT level from steam generatin
+    2      2      -0.0261    0.0818   0.1079   3698  2  FACTORY ng Co , also owns part of the PLANT , but S&P said its current
+    2      2      -0.0268    0.0049   0.0316   3835  2  FACTORY         In a 1987 trial , six PLANT officials were convicted f
+    2      2      -0.0273    0.0338   0.0612   3921  2  FACTORY  include accidents within the PLANT as well as accidents invol
+    2      2      -0.0275    0.0386   0.0661   3693  2  FACTORY  to make full use of existing PLANT and equipment </S>
+    2      2      -0.0281    0.0483   0.0764   3923  2  FACTORY ent of India charged that the PLANT design was poor and that p
*    1      2      -0.0283    0.0572   0.0855   3706  1  LIVING   indigo requires 400 units of PLANT material to produce 1 unit
+    2      2      -0.0287    0.0028   0.0315   3892  2  FACTORY  pride at the dedication of a PLANT expansion recently , and t
+    2      2      -0.0304    0.0081   0.0385   3893  2  FACTORY  in the sale of the Las Vegas PLANT and related assets </S>
+    2      2      -0.0307    0.0142   0.0449   3854  2  FACTORY s said it bought the site , a PLANT in Vestal , New York , fro
+    2      2      -0.0321    0.0465   0.0787   3791  2  FACTORY rs in a garment manufacturing PLANT </S>
+    2      2      -0.0344    0.0219   0.0563   3615  2  FACTORY by occupations already in the PLANT and unions are interested
+    2      2      -0.0350    0.0279   0.0629   3671  2  FACTORY ontrol of activities " at the PLANT near Oswego , N.Y </S>
*    1      2      -0.0353    0.0457   0.0809   3928  1  LIVING  an also endanger little-known PLANT and animal species </S>
+    2      2      -0.0355    0.0234   0.0589   3815  2  FACTORY vestigation of a death at the PLANT last April </S>
+    2      2      -0.0358    0.0081   0.0439   3783  2  FACTORY ll build a gallium extraction PLANT at its aluminum smelter in
+    2      2      -0.0363    0.0337   0.0700   3940  2  FACTORY                           The PLANT will convert naphtha and p
+    2      2      -0.0363    0.0254   0.0617   3704  2  FACTORY distinction between permanent PLANT closings and short-term ,
+    2      2      -0.0365    0.0522   0.0887   3991  2  FACTORY oduction in 1989 and increase PLANT capacity 20 % </S>
+    2      2      -0.0366    0.0306   0.0672   3729  2  FACTORY escribed in the large body of PLANT shutdown literature , but
+    2      2      -0.0383    0.0169   0.0552   3814  2  FACTORY that the NRC confirmed at the PLANT , but he said more than on
+    2      2      -0.0387    0.0501   0.0888   3622  2  FACTORY food concern , indicates that PLANT gene-splicing technology i
+    2      2      -0.0399    0.0034   0.0433   3777  2  FACTORY l Genstar 's gypsum wallboard PLANT and gypsum quarry at Las V
+    2      2      -0.0410    0.0424   0.0834   3664  2  FACTORY rage hourly production rate , PLANT capacity and shipping meth
+    2      2      -0.0411    0.0274   0.0685   3699  2  FACTORY oline component at its Botlek PLANT near Rotterdam </S>
+    2      2      -0.0429    0.0240   0.0670   3932  2  FACTORY which doesn't link the Hudson PLANT problem with any particula
+    2      2      -0.0433    0.0054   0.0487   3818  2  FACTORY nd closed its 16-mm projector PLANT in 1985 </S>
+    2      2      -0.0439    0.0310   0.0749   3884  2  FACTORY s will be used to improve its PLANT and properties </S>
+    2      2      -0.0444    0.0534   0.0978   3742  2  FACTORY  parts currently built at the PLANT will be discontinued , but
+    2      2      -0.0445    0.0591   0.1037   3768  2  FACTORY ibution to the study of power PLANT siting as a political prob
+    2      2      -0.0468    0.0633   0.1101   3702  2  FACTORY l administrator also said the PLANT 's problems " demonstrate
+    2      2      -0.0470    0.0312   0.0782   3672  2  FACTORY  example , the Racine tractor PLANT would operate with fewer t
+    2      2      -0.0476    0.0114   0.0590   3784  2  FACTORY  WHEELER CORP.'S cogeneration PLANT at Mt Carmel , Pa , will b
+    2      2      -0.0479    0.0476   0.0955   3638  2  FACTORY he program ; and at the third PLANT workers first supported th
+    2      2      -0.0500    0.0233   0.0734   3860  2  FACTORY               The Rancho Seco PLANT started up in 1974 , feedi
+    2      2      -0.0501    0.0300   0.0801   3862  2  FACTORY ving the railroad 's physical PLANT and making severance payme
+    2      2      -0.0509    0.0121   0.0630   3806  2  FACTORY eptable representation of the PLANT 's operations </S>
+    2      2      -0.0514    0.0884   0.1399   3628  2  FACTORY ers and former workers at the PLANT -- and at a few other plan
+    2      2      -0.0539    0.0146   0.0685   3617  2  FACTORY t at GE 's Greenville , S.C , PLANT , with shipment set for mi
+    2      2      -0.0549    0.0253   0.0802   3920  2  FACTORY States in 1954 at the Detroit PLANT of the McLouth Steel Corpo
+    2      2      -0.0554    0.0237   0.0791   3813  2  FACTORY he work force at its Evendale PLANT by 500 to 1,000 people </S
+    2      2      -0.0567    0.0058   0.0624   3907  2  FACTORY s , and a hydroelectric power PLANT has been constructed at th
+    2      2      -0.0570    0.0049   0.0619   4000  2  FACTORY ed to sell a gypsum wallboard PLANT and gypsum quarry near Las
+    2      2      -0.0616    0.0127   0.0742   3885  2  FACTORY otiating to sell or lease the PLANT to a group of Scottish bus
+    2      2      -0.0644    0.0088   0.0732   3697  2  FACTORY  workers at the Hyundai Motor PLANT in Ulsan ; workers giving
+    2      2      -0.0666    0.0240   0.0906   3836  2  FACTORY              In a modern coal PLANT the combustion of one poun
+    2      2      -0.0671    0.0207   0.0878   3939  2  FACTORY  to a proposed electric power PLANT in Texas </S>
+    2      2      -0.0676    0.0468   0.1144   3889  2  FACTORY ford , N.Y , silicon-products PLANT </S>
+    2      2      -0.0678    0.0259   0.0936   3626  2  FACTORY y chemicals and salt said the PLANT is expected to be under co
+    2      2      -0.0682    0.0212   0.0893   3739  2  FACTORY may still bid for work on the PLANT , they fear they won't be
+    2      2      -0.0683    0.0117   0.0801   3819  2  FACTORY loans needed for the Monessen PLANT 's electric furnace </S>
+    2      2      -0.0688    0.0380   0.1067   3623  2  FACTORY est to visit an Owens-Corning PLANT in Tennessee and on the tr
+    2      2      -0.0688    0.0132   0.0820   3787  2  FACTORY the workers ' entrance to the PLANT stadium , where rallies ar
+    2      2      -0.0694    0.0133   0.0826   3618  2  FACTORY sodium-borohydride production PLANT in Delfzijl , The Netherla
+    2      2      -0.0695    0.0334   0.1028   3999  2  FACTORY s at its New Bedford , Mass , PLANT </S>
+    2      2      -0.0696    0.0238   0.0933   3616  2  FACTORY ike houses , collectibles and PLANT equipment " </S>
+    2      2      -0.0703    0.0210   0.0913   3658  2  FACTORY an incident in 1985 -- when a PLANT operator mistakenly starte
+    2      2      -0.0717    0.0208   0.0926   3788  2  FACTORY  of a 1986 uranium-processing PLANT accident </S>
+    2      2      -0.0725    0.0191   0.0916   3903  2  FACTORY kers to regain control of the PLANT </S>
+    2      2      -0.0726    0.0345   0.1071   3620  2  FACTORY  and a natural gas processing PLANT near Calgary , Alberta , f
+    2      2      -0.0736    0.0156   0.0892   3669  2  FACTORY in its search for a new steel PLANT site </S>
+    2      2      -0.0736    0.0262   0.0998   3822  2  FACTORY s from its Camden offices and PLANT use its fitness center , d
+    2      2      -0.0738    0.0120   0.0857   3786  2  FACTORY  Firestone tire manufacturing PLANT in Salinas , Calif </S>
+    2      2      -0.0739    0.0353   0.1092   3821  2  FACTORY dy on a superconducting power PLANT and plans to have a workin
+    2      2      -0.0741    0.0045   0.0785   3774  2  FACTORY d a new liquids manufacturing PLANT in Cambridge , Ohio </S>
+    2      2      -0.0741    0.0334   0.1076   3625  2  FACTORY cided to build a new chemical PLANT in Saukville , Wis , after
+    2      2      -0.0742    0.0065   0.0807   3703  2  FACTORY ible strike today at its Jeep PLANT in Toledo , Ohio , was unc
+    2      2      -0.0746    0.0125   0.0871   3890  2  FACTORY anks were being heated at the PLANT , when , in fact , managem
+    2      2      -0.0755    0.0101   0.0857   3865  2  FACTORY ls Corp unit , which owns the PLANT , paid the NRC a $310,000
+    2      2      -0.0759    0.0194   0.0953   3997  2  FACTORY aft , La , phosphate-chemical PLANT and its inventory from Fre
+    2      2      -0.0763    0.0395   0.1159   3662  2  FACTORY            The gas processing PLANT , which was recently compl
+    2      2      -0.0764    0.0336   0.1100   3778  2  FACTORY ing the impact of the nuclear PLANT delay and said it expects
+    2      2      -0.0764    0.0247   0.1012   3936  2  FACTORY ll begin in early 1988 on the PLANT , to have production capac
+    2      2      -0.0765    0.0301   0.1066   3857  2  FACTORY Board that require disallowed PLANT costs to be treated as a r
+    2      2      -0.0778    0.0680   0.1459   3741  2  FACTORY too early to determine if the PLANT closing will result in a c
+    2      2      -0.0782    0.0053   0.0834   3665  2  FACTORY r hardwood mills and a veneer PLANT </S>
+    2      2      -0.0791    0.0293   0.1084   3882  2  FACTORY the Heber demonstration power PLANT in California , which is s
+    2      2      -0.0798    0.0336   0.1134   3909  2  FACTORY ts car , the De Lorean , at a PLANT in Belfast , Northern Irel
+    2      2      -0.0798    0.0175   0.0973   3690  2  FACTORY       The first nuclear power PLANT was placed aboard the subm
+    2      2      -0.0801    0.0251   0.1053   3660  2  FACTORY  engineers has scoured Newark PLANT records going back four de
+    2      2      -0.0801    0.0892   0.1693   3780  2  FACTORY ditions , and for a completed PLANT if certain costs for the p
+    2      2      -0.0811    0.0372   0.1183   3775  2  FACTORY 985 that could have saved the PLANT , but that the union leade
+    2      2      -0.0827    0.0420   0.1247   3820  2  FACTORY                    Moreover , PLANT closings would increase th
+    2      2      -0.0836    0.0043   0.0879   3861  2  FACTORY ntract to operate a munitions PLANT at Shreveport , La </S>
+    2      2      -0.0849    0.0776   0.1625   3653  2  FACTORY vidual cost profiles for each PLANT and an industry supply sch
+    2      2      -0.0858    0.0197   0.1055   3740  2  FACTORY ts Belvidere , Ill , assembly PLANT </S>
+    2      2      -0.0858    0.0296   0.1155   3633  2  FACTORY 8 billion Waterford 3 nuclear PLANT north of New Orleans </S>
+    2      2      -0.0863    0.0095   0.0958   3934  2  FACTORY gency planning issues until a PLANT is substantially construct
+    2      2      -0.0863    0.0194   0.1058   3776  2  FACTORY  Campbell 's electrical power PLANT , says he used to drink a
+    2      2      -0.0865    0.0227   0.1091   3929  2  FACTORY                           The PLANT will be closed by the end
+    2      2      -0.0882    0.0184   0.1066   3894  2  FACTORY nt in the Millstone 3 nuclear PLANT in Waterford , Conn , that
+    2      2      -0.0886    0.0212   0.1097   3930  2  FACTORY already started to design the PLANT and that construction will
+    2      2      -0.0888    0.0167   0.1055   3901  2  FACTORY ure will build a $100 million PLANT in Bethlehem , Pa , to con
+    2      2      -0.0893    0.0117   0.1010   3621  2  FACTORY on of a liquefied-natural-gas PLANT , and Chase had hoped to r
+    2      2      -0.0896    0.0572   0.1467   3927  2  FACTORY of design , package filling , PLANT operation , and other serv
+    2      2      -0.0906    0.0217   0.1123   3811  2  FACTORY egarding costs connected with PLANT abandonments or rate disal
+    2      2      -0.0914    0.0727   0.1641   3656  2  FACTORY number of production cuts and PLANT closures industrywide in r
+    2      2      -0.0923    0.0107   0.1030   3898  2  FACTORY uy all surplus power from the PLANT </S>
+    2      2      -0.0927    0.0253   0.1180   3782  2  FACTORY ion 's Chernobyl atomic-power PLANT , which cut farm income ;
+    2      2      -0.0932    0.0244   0.1176   3998  2  FACTORY f commercial vehicles at a VW PLANT in Hanover </S>
+    2      2      -0.0933    0.0452   0.1385   3931  2  FACTORY  and one-time expenses from a PLANT closure and other cost-cut
+    2      2      -0.0942    0.0218   0.1160   3856  2  FACTORY its investment in two nuclear PLANT units </S>
+    2      2      -0.0969    0.0405   0.1374   3888  2  FACTORY y close its only U.S assembly PLANT for the first time in more
+    2      2      -0.0977    0.0165   0.1142   3863  2  FACTORY ker and the UAW locals at the PLANT broke off talks Friday , t
+    2      2      -0.1017    0.0408   0.1424   3902  2  FACTORY        The company 's largest PLANT investment in at least 10
+    2      2      -0.1044    0.0195   0.1239   3700  2  FACTORY t its Indianapolis electrical PLANT </S>
+    2      2      -0.1051    0.0393   0.1443   3891  2  FACTORY ing Heights , Mich , assembly PLANT for the week of April 20 <
+    2      2      -0.1060    0.0300   0.1360   3897  2  FACTORY canceled Zimmer nuclear power PLANT to a coal-fired plant is s
+    2      2      -0.1060    0.0300   0.1360   3900  2  FACTORY r power plant to a coal-fired PLANT is scheduled for completio
+    2      2      -0.1074    0.0067   0.1142   3743  2  FACTORY er facilities at the Monessen PLANT </S>
+    2      2      -0.1080    0.0056   0.1136   3619  2  FACTORY loney " the argument that the PLANT was operating , Mr Hall sa
+    2      2      -0.1091    0.0429   0.1520   3896  2  FACTORY  large coal-fueled generating PLANT that went into operation l
+    2      2      -0.1103    0.0333   0.1436   3899  2  FACTORY  refuses to accept this power PLANT " </S>
+    2      2      -0.1105    0.0331   0.1436   3705  2  FACTORY get cut for the Oklahoma City PLANT demanded by Firestone </S>
+    2      2      -0.1110    0.0091   0.1201   3668  2  FACTORY          Unit I of the Vogtle PLANT currently is operating at
+    2      2      -0.1118    0.0193   0.1311   3785  2  FACTORY to the Seabrook nuclear power PLANT just two miles north of Ma
+    2      2      -0.1156    0.0108   0.1264   3941  2  FACTORY                           The PLANT also assembles , under a c
+    2      2      -0.1188    0.0671   0.1859   3859  2  FACTORY  of $8.6 million related to a PLANT closing </S>
+    2      2      -0.1199    0.0602   0.1801   3816  2  FACTORY t to increase the cost of the PLANT , which is to be completed
+    2      2      -0.1225    0.0471   0.1696   3695  2  FACTORY rred costs of the Waterford 3 PLANT </S>
+    2      2      -0.1240    0.0071   0.1311   3631  2  FACTORY to 1,000 jobs at a jet-engine PLANT at Evendale , Ohio , over
+    2      2      -0.1244    0.0178   0.1422   3659  2  FACTORY r Co , project manager of the PLANT , said the company has to
+    2      2      -0.1247    0.0137   0.1384   3735  2  FACTORY er work from the Indianapolis PLANT </S>
+    2      2      -0.1266    0.0180   0.1445   3985  2  FACTORY onstructing an aircraft parts PLANT in Macon , Ga </S>
+    2      2      -0.1270    0.0066   0.1336   3781  2  FACTORY act for operating a munitions PLANT at Independence , Mo </S>
+    2      2      -0.1284    0.0291   0.1575   3886  2  FACTORY ent would drive nuclear power PLANT operators to look abroad f
+    2      2      -0.1286    0.0361   0.1647   3810  2  FACTORY ction for workers affected by PLANT closings </S>
+    2      2      -0.1288    0.0341   0.1630   3737  2  FACTORY ties to write off any nuclear PLANT costs they can't recover w
+    2      2      -0.1312    0.0118   0.1431   3666  2  FACTORY tor Corp said it will build a PLANT in Youngstown , Ohio , to
+    2      2      -0.1339    0.0243   0.1582   3670  2  FACTORY perating its own cogeneration PLANT , capable of producing 80
+    2      2      -0.1391    0.0161   0.1553   3864  2  FACTORY  workers dismissed because of PLANT closings , it sets up a $4
+    2      2      -0.1394    0.0114   0.1508   3895  2  FACTORY rkers , who have occupied the PLANT since Jan. 14 in an effort
+    2      2      -0.1401    0.0149   0.1550   3663  2  FACTORY upled with recently announced PLANT closings , will have " onl
+    2      2      -0.1468    0.0076   0.1544   3887  2  FACTORY ontract for an electric power PLANT in Mendota , Calif </S>
+    2      2      -0.1469    0.0099   0.1568   3701  2  FACTORY its Oklahoma City tire-making PLANT </S>
+    2      2      -0.1484    0.0362   0.1847   3937  2  FACTORY ndoned portion of the nuclear PLANT </S>
+    2      2      -0.1497    0.0049   0.1546   3624  2  FACTORY . 's Alvin W Vogtle I nuclear PLANT in Waynesboro , Ga </S>
+    2      2      -0.1545    0.0189   0.1734   3734  2  FACTORY maker said it would close the PLANT , which makes passenger-ca
+    2      2      -0.1582    0.0263   0.1845   3938  2  FACTORY cord a domestic nuclear power PLANT in its second-quarter sale
+    2      2      -0.1660    0.0143   0.1803   3632  2  FACTORY  as two years ago to have the PLANT in commercial operation by
+    2      2      -0.1684    0.0221   0.1905   3627  2  FACTORY to the Seabrook nuclear power PLANT are about $2.5 million </S
+    2      2      -0.1696    0.0272   0.1967   3855  2  FACTORY  its California joint-venture PLANT with Toyota Motor Corp bec
+    2      2      -0.1713    0.0136   0.1850   3933  2  FACTORY  , which includes an assembly PLANT and engine and stamping op
+    2      2      -0.1891    0.0320   0.2211   3636  2  FACTORY t the Three Mile Island power PLANT near Harrisburg brought in
+    2      2      -0.1969    0.0305   0.2275   3935  2  FACTORY ion of capacity at its engine PLANT in Anna , Ohio and said it
+    2      2      -0.2139    0.0480   0.2619   3779  2  FACTORY l have to make , assuming the PLANT starts operation early nex
================================================================================================================================

Variations on the Model

I set up the weights before entering it to the doc_vectors array by calling a get_weights function and weigh the tokens
accordingly based on weighting scheme the user chooses. I implemented all three different weighting schemes with default
having uniform wieghts for all tokens and for the custom weighting scheme I made all distance of 1 of the target word have
weight of 100 and distance of 2 weight of 50 and the rest weight of 1. I did this to put more importance on the words close
to the ambiguous target word. Using this I got a suprisingly similar accuracy to other position weighting schemes. Also
implemented adjacent-separate-LR to distinguish the left and right words to the target words to have more accuracy.
An example of part 2 output is shown bellow.

Enter Option: 2
Choose file (plant/tank/perplace): plant
Choose stemmed or unstemmed: stemmed
Choose position weighting (exponential/stepped/custom): exponential
Include LR adjacency model (yes/no): no
   sense  predict  sim1-sim2   sim1     sim2    doc#    sense    title
   
================================================================================================================================
+    1      1       0.2107    0.2197   0.0090   3953  1  LIVING  ) are essential nutrients for PLANT growth and development , p
+    1      1       0.2057    0.2387   0.0329   3792  1  LIVING  ld per plant , internodes and PLANT height were reduced by pla
+    1      1       0.2017    0.2229   0.0212   3926  1  LIVING  ntrolling growth of the whole PLANT </S>
+    1      1       0.1998    0.2159   0.0161   3948  1  LIVING  cted agronomic traits such as PLANT height , number of ear-bea
+    1      1       0.1958    0.1976   0.0018   3954  1  LIVING  n had greater seed weight and PLANT height than Hobbit and Asg
+    1      1       0.1930    0.2009   0.0080   3712  1  LIVING  rotein , 1000 kernel weight , PLANT height , kernels head$\sp
+    1      1       0.1864    0.2101   0.0237   3681  1  LIVING   N ) , and the effects of the PLANT growth retardant paclobutr
+    1      1       0.1828    0.1932   0.0104   3673  1  LIVING  d to merge cells of different PLANT species </S>
+    1      1       0.1799    0.1954   0.0155   3728  1  LIVING  , and the growth stage of the PLANT cells in the reaction mixt
+    1      1       0.1746    0.1875   0.0129   3830  1  LIVING  ssortment of animals and some PLANT material and are found nea
+    1      1       0.1740    0.1853   0.0113   3826  1  LIVING  in BS10 and BS11 mean ear and PLANT height decreased from the
+    1      1       0.1724    0.2012   0.0288   3751  1  LIVING  different disease ratings and PLANT height was non significant
+    1      1       0.1469    0.1622   0.0153   3745  1  LIVING                             As PLANT population increased , ear
+    1      1       0.1464    0.1684   0.0220   3842  1  LIVING  reater number of branches per PLANT , more pods per plant , fe
+    1      1       0.1463    0.1585   0.0123   3870  1  LIVING  species successfully promoted PLANT growth under highly fertil
+    1      1       0.1434    0.1737   0.0303   3720  1  LIVING  gronomic characters including PLANT height and days to 50 % si
+    1      1       0.1410    0.1578   0.0168   3769  1  LIVING   11.6 and 4.8 % reductions in PLANT height and 18.8 and 21.0 %
+    1      1       0.1406    0.1667   0.0261   3831  1  LIVING  hoot and leaflet size , total PLANT leaf area and in several r
+    1      1       0.1393    0.1724   0.0330   3960  1  LIVING  aters Sulfur occurs in living PLANT and animal tissue as part
+    1      1       0.1369    0.1411   0.0042   3825  1  LIVING  n of proteins that constitute PLANT and animal cells </S>
+    1      1       0.1332    0.1543   0.0211   3847  1  LIVING   potential application to all PLANT species of such CLONING of
+    1      1       0.1329    0.1458   0.0128   3879  1  LIVING  mine water stress in numerous PLANT species </S>
+    1      1       0.1309    0.1462   0.0153   3834  1  LIVING  charge resulting in decreased PLANT available soil water </S>
+    1      1       0.1304    0.1544   0.0240   3744  1  LIVING  of drying was observed in the PLANT grown in soil A than that
+    1      1       0.1294    0.1517   0.0223   3846  1  LIVING  tion achieved with the use of PLANT growth regulators </S>
+    1      1       0.1282    0.1729   0.0446   3904  1  LIVING  t improvement with increasing PLANT densities for three experi
+    1      1       0.1252    0.1369   0.0117   3914  1  LIVING  ium flux and shoot weight per PLANT indicated that selection f
+    1      1       0.1238    0.1381   0.0143   3726  1  LIVING  those substances derived from PLANT and animal ( organic ) sou
+    1      1       0.1204    0.1526   0.0322   3767  1  LIVING  ee of damage depending on the PLANT species and the time at wh
+    1      1       0.1202    0.1317   0.0115   3793  1  LIVING  eight in AS-D and lodging and PLANT appearance in AS-3 </S>
+    1      1       0.1198    0.1369   0.0171   3868  1  LIVING  ormonal control of growth The PLANT body of a hornwort , or ga
+    1      1       0.1193    0.1301   0.0108   3651  1  LIVING  tration at R6 , proportion of PLANT N from fixation , and seed
+    1      1       0.1182    0.1431   0.0250   3640  1  LIVING  tems used only two kingdoms , PLANT and animal , whereas most
+    1      1       0.1181    0.1709   0.0528   3797  1  LIVING  evels of various nutrients in PLANT parts </S>
+    1      1       0.1179    0.1511   0.0332   3718  1  LIVING  ce the growth and health of a PLANT , including such inherited
+    1      1       0.1174    0.1208   0.0034   3961  1  LIVING   number of seeds in NP21R and PLANT height and panicle length
+    1      1       0.1151    0.1355   0.0204   3771  1  LIVING  af , and number of leaves per PLANT ; however , P vulgaris pla
+    1      1       0.1150    0.1219   0.0069   3710  1  LIVING  s , leaves , or fruits of the PLANT </S>
+    1      1       0.1144    0.1313   0.0169   3867  1  LIVING  acing , resulting in variable PLANT density </S>
+    1      1       0.1140    0.1384   0.0245   3880  1  LIVING  cy , and importance values of PLANT species , including Trautv
+    1      1       0.1121    0.1451   0.0330   3750  1  LIVING  ount of leaf area , and total PLANT weight followed by sicklep
+    1      1       0.1112    0.1204   0.0092   3746  1  LIVING  on data , in conjunction with PLANT height data , allowed dete
+    1      1       0.1083    0.1158   0.0075   3612  1  LIVING  ( kg ) , dry weight ( % ) and PLANT volume ( m$\sp 3 $ ) </S>
+    1      1       0.1070    0.1454   0.0385   3928  1  LIVING  an also endanger little-known PLANT and animal species </S>
+    1      1       0.1066    0.1112   0.0046   3790  1  LIVING  he leaves to the grain during PLANT maturation </S>
+    1      1       0.1044    0.1219   0.0175   3809  1  LIVING  buted to reduced Mn levels in PLANT tissue </S>
+    1      1       0.1025    0.1045   0.0020   3717  1  LIVING  odule number , and nodule and PLANT dry weight </S>
+    1      1       0.1020    0.1205   0.0185   3919  1  LIVING  und , coverages of individual PLANT species , and screening co
+    1      1       0.1002    0.1261   0.0259   3758  1  LIVING  eatly stimulated the study of PLANT diseases Until 1900 , plan
+    1      1       0.0987    0.1094   0.0107   3828  1  LIVING  flowering the average sorghum PLANT had accumulated 63 % and 7
+    1      1       0.0976    0.1340   0.0364   3757  1  LIVING  f grains in many parts of the PLANT , principally in embryonic
+    1      1       0.0969    0.1024   0.0055   3883  1  LIVING  h H$\sp { + } $ released from PLANT roots </S>
+    1      1       0.0969    0.1371   0.0403   3759  1  LIVING  ment and the growth rate of a PLANT in isolation is a measure
+    1      1       0.0964    0.0977   0.0013   3602  1  LIVING  dine and deoxycytidine rescue PLANT growth and differentiation
+    1      1       0.0941    0.1031   0.0090   3755  1  LIVING  ithin the tissues of the host PLANT </S>
+    1      1       0.0935    0.1091   0.0155   3924  1  LIVING  econditioning water stress on PLANT morphology , yield compone
+    1      1       0.0903    0.0954   0.0051   3721  1  LIVING  o 0.75 m from the stem of the PLANT </S>
+    1      1       0.0886    0.1033   0.0147   3609  1  LIVING   bottoms for organic ( mostly PLANT ) material , which they st
+    1      1       0.0881    0.1123   0.0243   3912  1  LIVING  avity of seeds on emergence , PLANT characteristics , and yiel
+    1      1       0.0871    0.0991   0.0120   3839  1  LIVING  ds contain two of the primary PLANT nutrients and are called m
+    1      1       0.0868    0.1170   0.0302   3832  1  LIVING  cycling of minerals by eating PLANT parts and then excreting s
+    1      1       0.0868    0.0916   0.0048   3680  1  LIVING   In the low phosphorus soil , PLANT phosphorus uptake was incr
+    1      1       0.0824    0.1082   0.0258   3943  1  LIVING  g to avoid being eaten , many PLANT species have developed som
+    1      1       0.0816    0.0979   0.0163   3947  1  LIVING  ned included effects of : the PLANT hormone auxin ; buffers of
+    1      1       0.0812    0.1127   0.0314   3944  1  LIVING  er-soluble carbohydrate among PLANT parts at either 10 or 30C
+    1      1       0.0804    0.1116   0.0313   3761  1  LIVING  osphorus concentration in all PLANT parts </S>
+    1      1       0.0800    0.1019   0.0218   3682  1  LIVING  ve photosynthetic area of the PLANT </S>
+    1      1       0.0797    0.1092   0.0294   3674  1  LIVING               All parts of the PLANT , particularly the berries
+    1      1       0.0777    0.1034   0.0257   3713  1  LIVING   years were lower in soil and PLANT Zn than other fields </S>
+    1      1       0.0776    0.0903   0.0127   3707  1  LIVING  es that the cell or thallus ( PLANT body ) lacks : roots , ste
+    1      1       0.0775    0.0806   0.0032   3922  1  LIVING                     The cotton PLANT grows upright to a height
+    1      1       0.0769    0.1105   0.0337   3918  1  LIVING  evelopment and differences in PLANT size </S>
+    1      1       0.0755    0.0807   0.0052   3950  1  LIVING  spatial variability ( HH ) in PLANT species and forest structu
+    1      1       0.0745    0.1005   0.0260   3684  1  LIVING  fruit decay , reduced size of PLANT , matured sooner and deter
*    2      1       0.0744    0.1174   0.0429   3996  2  FACTORY                     The total PLANT population consisted of 24
+    1      1       0.0738    0.0890   0.0152   3789  1  LIVING  fa yield when S levels in the PLANT are below 2.5 mg S ( 0.25
+    1      1       0.0728    0.0774   0.0046   3878  1  LIVING  essive epiphytic colonizer of PLANT tissues </S>
+    1      1       0.0725    0.1046   0.0320   3906  1  LIVING  morphological position on the PLANT was developed by which 65
+    1      1       0.0717    0.0954   0.0236   3763  1  LIVING   1 ) determine the effects of PLANT cover and trampling by liv
+    1      1       0.0708    0.0712   0.0004   3605  1  LIVING  ipened ovary of any flowering PLANT , or ANGIOSPERM , and usua
+    1      1       0.0702    0.0719   0.0018   3642  1  LIVING  ed cheat spike number , cheat PLANT dry weight , and cheat see
+    1      1       0.0699    0.1081   0.0381   3708  1  LIVING  t experiment evaluated single PLANT response to treatment and
+    1      1       0.0693    0.0777   0.0084   3959  1  LIVING  itivity to APAR under varying PLANT and soil moisture conditio
+    1      1       0.0686    0.1296   0.0610   3987  1  LIVING  er " as an integral part of a PLANT </S>
+    1      1       0.0677    0.0719   0.0042   3764  1  LIVING  trong correlation between the PLANT characteristics that were
+    1      1       0.0672    0.1058   0.0387   3722  1  LIVING   production with no effect on PLANT vigor </S>
+    1      1       0.0662    0.0878   0.0216   3634  1  LIVING  zation of nitrogen within the PLANT occurs during periods of l
+    1      1       0.0660    0.0736   0.0076   3796  1  LIVING  carved or etched naturalistic PLANT and animal forms </S>
+    1      1       0.0660    0.0725   0.0065   3756  1  LIVING  two N levels ( 40 or 200 mg N PLANT ( '-1 ) ) and four daytime
+    1      1       0.0657    0.1026   0.0369   3876  1  LIVING  opulation with a greater mean PLANT size and reduced coefficie
+    1      1       0.0656    0.0952   0.0296   3752  1  LIVING                  In most other PLANT groups the leaves enlarge
+    1      1       0.0655    0.0768   0.0113   3917  1  LIVING            Results showed that PLANT roots contain AcP , NP , a
+    1      1       0.0654    0.0710   0.0056   3852  1  LIVING  tly associated with decreased PLANT vigor </S>
+    1      1       0.0649    0.0757   0.0108   3905  1  LIVING  ad penetrated walls of living PLANT cells and ramified intra -
+    1      1       0.0642    0.0792   0.0149   3875  1  LIVING  rry ( Vaccinium ashei Reade ) PLANT survival , growth , develo
+    1      1       0.0630    0.0664   0.0034   3754  1  LIVING  0$\sp5 $ g$\sp { -1 } $ fresh PLANT </S>
+    1      1       0.0624    0.0907   0.0283   3749  1  LIVING  d more than merely supplement PLANT foods with animal products
+    1      1       0.0616    0.0743   0.0127   3910  1  LIVING  In 1979 , pitted morningglory PLANT weights were reduced by 66
+    1      1       0.0575    0.0965   0.0390   3641  1  LIVING  an cell wall thickness of the PLANT tissues than to any of the
+    1      1       0.0543    0.1054   0.0510   3732  1  LIVING  sect resistance , and general PLANT and line appearance </S>
+    1      1       0.0536    0.0890   0.0354   3824  1  LIVING  e short term had no effect on PLANT survival , two consecutive
+    1      1       0.0536    0.1030   0.0494   3848  1  LIVING  e quantities of wood or other PLANT materials ( for a time , k
+    1      1       0.0520    0.0731   0.0212   3977  1  LIVING   they have been troubled with PLANT diseases </S>
+    1      1       0.0514    0.0724   0.0210   3946  1  LIVING                      Efficient PLANT regeneration was obtained
+    1      1       0.0499    0.0589   0.0090   3794  1  LIVING        Grafting is a method of PLANT propagation in which a sci
+    1      1       0.0496    0.0582   0.0086   3807  1  LIVING   influence of the nematode on PLANT hose physiology might be p
+    1      1       0.0489    0.0671   0.0183   3978  1  LIVING  shed as vital repositories of PLANT seeds ; a large genetic po
+    1      1       0.0478    0.0516   0.0038   3677  1  LIVING   , C canadensis--a herbaceous PLANT , with woody rootstock , w
+    1      1       0.0471    0.0938   0.0467   3753  1  LIVING   function , soil physical and PLANT physiological aspects are
+    1      1       0.0461    0.0572   0.0111   3808  1  LIVING  ating a crucial role for this PLANT organ in the perception of
*    2      1       0.0460    0.1084   0.0624   3967  2  FACTORY rm products and soil near the PLANT have almost returned to no
+    1      1       0.0453    0.0471   0.0018   3799  1  LIVING  us esculentus , is a tropical PLANT grown as a vegetable </S>
+    1      1       0.0445    0.0648   0.0203   3993  1  LIVING  s or eventually recycled into PLANT matter </S>
+    1      1       0.0439    0.0624   0.0185   3872  1  LIVING  reduced small yellow nutsedge PLANT size by more than 50 % </S
+    1      1       0.0435    0.0477   0.0042   3837  1  LIVING  the bast fibers from the flax PLANT </S>
+    1      1       0.0434    0.0556   0.0122   3679  1  LIVING   did not reverse to untreated PLANT levels with an IE and was
+    1      1       0.0431    0.0587   0.0155   3715  1  LIVING  th traits in order to aid the PLANT breeder </S>
+    1      1       0.0424    0.0452   0.0028   3637  1  LIVING  s for resistance to tarnished PLANT bug </S>
+    1      1       0.0421    0.0608   0.0188   3795  1  LIVING   show relationships among the PLANT communities and to identif
+    1      1       0.0401    0.0452   0.0051   3833  1  LIVING          Heritability of these PLANT parameters which could be
+    1      1       0.0397    0.0531   0.0134   3986  1  LIVING   human deaths attributable to PLANT poisons is insignificant <
+    1      1       0.0385    0.0881   0.0495   3976  1  LIVING  of the seed of new , improved PLANT varieties </S>
+    1      1       0.0383    0.0650   0.0266   3725  1  LIVING  y be used to repair a type of PLANT damage known as girdling ,
+    1      1       0.0371    0.0512   0.0140   3686  1  LIVING         Research revealed that PLANT extracts had little hypogl
+    1      1       0.0371    0.0444   0.0073   3601  1  LIVING  eaflets sampled from the same PLANT </S>
+    1      1       0.0371    0.0487   0.0117   3613  1  LIVING            The relationship of PLANT and weather measurements a
+    1      1       0.0365    0.0680   0.0315   3990  1  LIVING  most unlimited potential that PLANT breeding provides , and th
+    1      1       0.0348    0.0479   0.0131   3611  1  LIVING  uence of the inability of the PLANT to extract 100 % of the wa
+    1      1       0.0348    0.0640   0.0292   3925  1  LIVING  edures that constitute modern PLANT breeding </S>
*    2      1       0.0344    0.1188   0.0844   3698  2  FACTORY ng Co , also owns part of the PLANT , but S&P said its current
+    1      1       0.0343    0.0534   0.0191   3803  1  LIVING  st-size leaf particle The tea PLANT found in Taiwan and south
+    1      1       0.0340    0.0407   0.0067   3851  1  LIVING  presence of natural tarnished PLANT bug infestations </S>
+    1      1       0.0337    0.0393   0.0055   3915  1  LIVING   : pollen is carried from one PLANT to another by wind or inse
+    1      1       0.0331    0.0487   0.0155   3675  1  LIVING  the riparian and non-riparian PLANT communities </S>
+    1      1       0.0328    0.0719   0.0392   3765  1  LIVING  viously ungrazed portion of a PLANT than the part which had be
+    1      1       0.0326    0.0410   0.0085   3711  1  LIVING  ound to be poor predictors of PLANT preferences except in tria
+    1      1       0.0325    0.0347   0.0022   3770  1  LIVING  ph Botrytis fabae on fhe host PLANT Vicia faba were studied </
+    1      1       0.0324    0.0429   0.0106   3685  1  LIVING  atpea changed with respect to PLANT organs and experimental fa
+    1      1       0.0322    0.0354   0.0032   3731  1  LIVING                 Three distinct PLANT viruses , transmitted by t
+    1      1       0.0316    0.0708   0.0392   3952  1  LIVING   be continuous throughout the PLANT </S>
+    1      1       0.0310    0.0446   0.0136   3647  1  LIVING  itutes substantive proof that PLANT breeding was practiced by
+    1      1       0.0307    0.0549   0.0242   3843  1  LIVING  d when 1 of the stems on each PLANT were clipped at each cut <
+    1      1       0.0303    0.0457   0.0154   3800  1  LIVING  ber of the potyvirus group of PLANT viruses </S>
+    1      1       0.0302    0.0407   0.0104   3874  1  LIVING  s , and many other xerophytic PLANT foods </S>
+    1      1       0.0302    0.0584   0.0282   3716  1  LIVING  corn stand using the isolated PLANT as a model </S>
+    1      1       0.0301    0.0498   0.0197   3873  1  LIVING  re low in nutrients The onion PLANT is potentially a biennial
+    1      1       0.0300    0.0643   0.0343   3983  1  LIVING  se have been developed in the PLANT as a defense against preda
+    1      1       0.0300    0.0365   0.0065   3646  1  LIVING   , sea spray , bacteria , and PLANT spores </S>
+    1      1       0.0281    0.0349   0.0068   3766  1  LIVING  s of the C ( ,3 ) terrestrial PLANT , Asparagus sprengeri Rege
+    1      1       0.0281    0.0690   0.0409   3773  1  LIVING  mpounds are a large family of PLANT chemicals composed of a su
+    1      1       0.0276    0.0485   0.0209   3966  1  LIVING  al release , is maintained in PLANT breeding </S>
+    1      1       0.0266    0.0317   0.0051   3635  1  LIVING  igands to immobilize isolated PLANT RNA polymerase II and asso
+    1      1       0.0265    0.0439   0.0174   3606  1  LIVING  pening phase , a mechanism of PLANT bug tolerance </S>
+    1      1       0.0264    0.0553   0.0289   3942  1  LIVING  ation has been limited by the PLANT 's vulnerability to virus
+    1      1       0.0260    0.0436   0.0176   3968  1  LIVING  ceptions of the importance of PLANT science competencies from
+    1      1       0.0257    0.0293   0.0035   3607  1  LIVING   and K8 had fewer nodules per PLANT than Cunningham and K67 bu
+    1      1       0.0256    0.0355   0.0099   3827  1  LIVING                      Among the PLANT eaters were a type of hors
+    1      1       0.0256    0.0627   0.0371   3845  1  LIVING  ween soil taxonomic units and PLANT associations ; and discrim
+    1      1       0.0255    0.0553   0.0298   3692  1  LIVING  vantages in converting scarce PLANT resources into food </S>
+    1      1       0.0254    0.0332   0.0078   3762  1  LIVING  ction of nodABC expression by PLANT flavonoids , and probably
+    1      1       0.0253    0.0392   0.0139   3723  1  LIVING  n various parts of the potato PLANT was quantitated by rocket
+    1      1       0.0251    0.0304   0.0052   3871  1  LIVING   is one of the weakest of the PLANT fibers , its use is limite
+    1      1       0.0248    0.0382   0.0134   3913  1  LIVING  ural carbohydrate ( NSC ) per PLANT but did influence NSC part
*    2      1       0.0247    0.0474   0.0227   3970  2  FACTORY ome supplies from the Spanish PLANT , an acid-like substance n
+    1      1       0.0238    0.0336   0.0097   3802  1  LIVING  and the habitat of the parent PLANT can often be inferred from
+    1      1       0.0234    0.0615   0.0381   3709  1  LIVING  lk position were utilized for PLANT samples </S>
*    2      1       0.0232    0.0870   0.0638   3724  2  FACTORY e one Quality of Working Life PLANT </S>
+    1      1       0.0227    0.0417   0.0190   3645  1  LIVING   important predictor of final PLANT uplift , was positively co
+    1      1       0.0217    0.0269   0.0052   3639  1  LIVING                  Derris is any PLANT of the genus Derris in the
*    2      1       0.0211    0.0513   0.0302   3736  2  FACTORY        " Several parts of the PLANT are fairly hazardous , inc
+    1      1       0.0210    0.0396   0.0186   3603  1  LIVING  th 300 ppm N Foliar color and PLANT quality was highest with 3
+    1      1       0.0209    0.0313   0.0104   3644  1  LIVING  orted for vertebrate and some PLANT mitochondria , and lack of
+    1      1       0.0209    0.0279   0.0070   3911  1  LIVING                      In higher PLANT membranes , lecithin and P
+    1      1       0.0208    0.0210   0.0002   3654  1  LIVING  uelen cabbage is a herbaceous PLANT , Pringlea antiscorbutica
+    1      1       0.0204    0.0546   0.0342   3691  1  LIVING            The general goal of PLANT breeding is to assemble in
+    1      1       0.0200    0.0270   0.0070   3687  1  LIVING                            The PLANT 's essential oil is used i
+    1      1       0.0199    0.0298   0.0099   3908  1  LIVING  or Theresa Arnold , the agave PLANT has special significance a
+    1      1       0.0199    0.0495   0.0297   3678  1  LIVING  ed more than 60 % of the same PLANT communities each season ,
*    2      1       0.0196    0.0884   0.0688   3604  2  FACTORY                      For most PLANT tissues , construction cos
+    1      1       0.0195    0.0471   0.0276   3823  1  LIVING  yford scientists identified a PLANT gene that produces " true
+    1      1       0.0183    0.0329   0.0146   3956  1  LIVING                     Ranking of PLANT associates and environment
*    2      1       0.0181    0.0264   0.0083   3988  2  FACTORY                           The PLANT used a lacquer coating on
+    1      1       0.0179    0.0336   0.0156   3760  1  LIVING  nts derived from the FOXGLOVE PLANT , including digitoxin , th
+    1      1       0.0176    0.0241   0.0065   3804  1  LIVING                     The entire PLANT is usually pure white , or
+    1      1       0.0169    0.0319   0.0150   3727  1  LIVING  is a tall , tropical , annual PLANT in the caper family , Capp
+    1      1       0.0163    0.0263   0.0100   3689  1  LIVING   Rieske iron-sulfur center in PLANT mitochondria , and a diffe
+    1      1       0.0156    0.0290   0.0134   3608  1  LIVING  ometimes known as the tapioca PLANT , is a member of the spurg
+    1      1       0.0136    0.0386   0.0250   3975  1  LIVING  basis for his hypothesis that PLANT domestication and early fa
+    1      1       0.0133    0.0194   0.0060   3849  1  LIVING                           This PLANT is native to Colorado and
+    1      1       0.0133    0.0260   0.0127   3965  1  LIVING  M ) was used to analyze pilot PLANT experimental data </S>
+    1      1       0.0132    0.0299   0.0168   3614  1  LIVING                   The story of PLANT breeding since 1920 or so
*    2      1       0.0129    0.0405   0.0276   3801  2  FACTORY  margin , debt ratio , age of PLANT , and current ratio </S>
*    2      1       0.0126    0.0222   0.0096   3747  2  FACTORY          Change occurs at the PLANT ( speculation ) , or at th
+    1      1       0.0125    0.0357   0.0232   3850  1  LIVING  ncountered by geneticists and PLANT breeders working with cott
+    1      1       0.0124    0.0190   0.0067   3963  1  LIVING   , however , cross reacted to PLANT antigens and thus are unsu
+    1      1       0.0118    0.0183   0.0066   3916  1  LIVING  ent in diploid and tetraploid PLANT sections were supportive o
+    1      1       0.0116    0.0254   0.0138   3982  1  LIVING  ers that collect and preserve PLANT germ plasm </S>
+    1      1       0.0108    0.1047   0.0939   3706  1  LIVING   indigo requires 400 units of PLANT material to produce 1 unit
+    1      1       0.0107    0.0304   0.0198   3838  1  LIVING  antisera raised against whole PLANT protein antigens </S>
+    1      1       0.0104    0.0135   0.0030   3719  1  LIVING            Nitrogen loading as PLANT detritus into hyacinth pon
+    1      1       0.0104    0.0413   0.0309   3958  1  LIVING  -stem portion of the 1-yr-old PLANT and favored quinine produc
+    1      1       0.0096    0.0155   0.0059   3610  1  LIVING   in mixed borders or as a pot PLANT 0277750-0 Stonehenge </S>
+    1      1       0.0095    0.0174   0.0080   3994  1  LIVING  The prey trips a lever on the PLANT " door , " which allows wa
+    1      1       0.0092    0.0274   0.0182   3869  1  LIVING  onditions as similar existing PLANT assemblages do today </S>
*    2      1       0.0087    0.0181   0.0094   3969  2  FACTORY                   A large MSF PLANT may have 60 flash chambers
+    1      1       0.0080    0.0240   0.0160   3688  1  LIVING                            The PLANT compartment is considered
+    1      1       0.0077    0.0119   0.0042   3881  1  LIVING  rly mildew , Oidium , and the PLANT louse , Phylloxera </S>
+    1      1       0.0074    0.0474   0.0399   3733  1  LIVING  water from a nylon processing PLANT was applied to 'Ky 31 ' ta
*    2      1       0.0073    0.1032   0.0959   3989  2  FACTORY                           The PLANT will produce parts for the
*    2      1       0.0068    0.0534   0.0466   3683  2  FACTORY  the grain volume than single PLANT firms </S>
+    1      1       0.0066    0.0183   0.0117   3841  1  LIVING  gus or pickled , or the whole PLANT ground and made into mush
+    1      1       0.0061    0.0328   0.0266   3951  1  LIVING         In extreme cases the " PLANT " may actually develop int
*    2      1       0.0055    0.0317   0.0262   3973  2  FACTORY  of such maneuvers , the main PLANT has experienced a parts sh
*    2      1       0.0035    0.0569   0.0533   3730  2  FACTORY hm using data measured at the PLANT level from steam generatin
*    2      1       0.0027    0.0121   0.0095   3714  2  FACTORY lso investigated explosives , PLANT chemistry , and the histor
*    2      1       0.0018    0.0145   0.0127   3853  2  FACTORY  addition of a Ralston Purina PLANT , it became the world 's b
+    1      1       0.0015    0.0129   0.0115   3748  1  LIVING  corated with brightly painted PLANT motifs , especially papyru
*    2      1       0.0008    0.0244   0.0237   3962  2  FACTORY nt light tones--such as the " PLANT light ' that mimics sunlig
+    1      1       0.0007    0.0438   0.0431   3643  1  LIVING   was transferred to a tobacco PLANT </S>
*    2      1       0.0005    0.0106   0.0101   3652  2  FACTORY tant limits are viewed by the PLANT as prima facie evidence of
*    1      2      -0.0002    0.0642   0.0644   3772  1  LIVING  ontaining natural products of PLANT origin that have an alkali
*    1      2      -0.0003    0.0185   0.0188   3964  1  LIVING  te the carbon influx into the PLANT </S>
+    2      2      -0.0011    0.0125   0.0136   3648  2  FACTORY e Oak Ridge gaseous diffusion PLANT </S>
*    1      2      -0.0018    0.0148   0.0166   3981  1  LIVING  loration Hyssop is an ancient PLANT and has been considered sa
+    2      2      -0.0018    0.0034   0.0052   3667  2  FACTORY t a coroner 's inquest at the PLANT this morning </S>
+    2      2      -0.0022    0.0291   0.0313   3979  2  FACTORY oes on : " But our Fort Myers PLANT , because of the warm wate
+    2      2      -0.0025    0.0163   0.0188   3805  2  FACTORY ls in bleaching and the pilot PLANT trial in thermomechanical
+    2      2      -0.0026    0.0508   0.0534   3622  2  FACTORY food concern , indicates that PLANT gene-splicing technology i
+    2      2      -0.0030    0.0768   0.0798   3877  2  FACTORY ngle product may occur at the PLANT in each time period </S>
*    1      2      -0.0030    0.0116   0.0147   3974  1  LIVING               Besides offering PLANT displays , many have botan
*    1      2      -0.0036    0.0408   0.0444   3676  1  LIVING  age utilization or for future PLANT improvement </S>
+    2      2      -0.0036    0.0085   0.0122   3955  2  FACTORY icinity of a titanium dioxide PLANT in the Gulf of Bothnia </S
*    1      2      -0.0043    0.0405   0.0447   3992  1  LIVING  h facilities of importance in PLANT taxonomy </S>
+    2      2      -0.0046    0.0002   0.0048   3738  2  FACTORY Metros at Rover 's Longbridge PLANT near Birmingham , is one o
*    1      2      -0.0060    0.0419   0.0478   3984  1  LIVING  mpotence , and so long as the PLANT remained rare in Europe ,
*    1      2      -0.0065    0.0363   0.0427   3957  1  LIVING  se it was once extracted from PLANT ashes ; now almost all is
+    2      2      -0.0071    0.0194   0.0265   3812  2  FACTORY n fish sticks in a processing PLANT , they can't sound like st
+    2      2      -0.0072    0.0309   0.0380   3630  2  FACTORY nd told us a sewage-treatment PLANT was going in " </S>
+    2      2      -0.0073    0.0236   0.0309   3696  2  FACTORY al director , said the Newark PLANT had frequently been used t
*    1      2      -0.0074    0.0342   0.0416   3840  1  LIVING                            The PLANT remains in bloom for about
+    2      2      -0.0074    0.0132   0.0206   3699  2  FACTORY oline component at its Botlek PLANT near Rotterdam </S>
*    1      2      -0.0086    0.0168   0.0254   3694  1  LIVING   used it to make a " pomato " PLANT , although its tomatoes an
*    1      2      -0.0093    0.0109   0.0202   3980  1  LIVING  boldt laid the foundations of PLANT geography , performing fie
+    2      2      -0.0100    0.0319   0.0420   3884  2  FACTORY s will be used to improve its PLANT and properties </S>
+    2      2      -0.0117    0.0225   0.0341   3815  2  FACTORY vestigation of a death at the PLANT last April </S>
+    2      2      -0.0119    0.0190   0.0309   3971  2  FACTORY           And like the Soviet PLANT , it lacks a thick , concr
+    2      2      -0.0129    0.0081   0.0210   3817  2  FACTORY of improper procedures at the PLANT and allowed them to contin
+    2      2      -0.0135    0.0037   0.0172   3858  2  FACTORY heir delivery to the Winnipeg PLANT would be delayed because o
+    2      2      -0.0150    0.0172   0.0322   3649  2  FACTORY s who , in turn , shipped the PLANT to Europe as a high-priced
+    2      2      -0.0152    0.0151   0.0304   3814  2  FACTORY that the NRC confirmed at the PLANT , but he said more than on
*    1      2      -0.0153    0.0414   0.0566   3995  1  LIVING  assess the impact of physical PLANT deferred maintenance , and
+    2      2      -0.0155    0.0175   0.0329   3783  2  FACTORY ll build a gallium extraction PLANT at its aluminum smelter in
+    2      2      -0.0157    0.0011   0.0168   3777  2  FACTORY l Genstar 's gypsum wallboard PLANT and gypsum quarry at Las V
+    2      2      -0.0158    0.0314   0.0472   3623  2  FACTORY est to visit an Owens-Corning PLANT in Tennessee and on the tr
+    2      2      -0.0167    0.0281   0.0448   3671  2  FACTORY ontrol of activities " at the PLANT near Oswego , N.Y </S>
+    2      2      -0.0167    0.0137   0.0305   3854  2  FACTORY s said it bought the site , a PLANT in Vestal , New York , fro
+    2      2      -0.0169    0.0013   0.0182   4000  2  FACTORY ed to sell a gypsum wallboard PLANT and gypsum quarry near Las
+    2      2      -0.0170    0.0080   0.0249   3657  2  FACTORY rchild 's Long Island , N.Y , PLANT , emanates from New York l
*    1      2      -0.0170    0.0467   0.0637   3945  1  LIVING  ion experiments , making this PLANT a potential virus reservoi
+    2      2      -0.0178    0.0225   0.0403   3885  2  FACTORY otiating to sell or lease the PLANT to a group of Scottish bus
+    2      2      -0.0189    0.0200   0.0389   3655  2  FACTORY  , 1981 , nine days after the PLANT 's license was revoked </S
+    2      2      -0.0190    0.0062   0.0252   3998  2  FACTORY f commercial vehicles at a VW PLANT in Hanover </S>
+    2      2      -0.0192    0.0198   0.0390   3940  2  FACTORY                           The PLANT will convert naphtha and p
+    2      2      -0.0218    0.0077   0.0295   3835  2  FACTORY         In a 1987 trial , six PLANT officials were convicted f
+    2      2      -0.0227    0.0419   0.0646   3650  2  FACTORY n enrollment and the physical PLANT while financial problems w
+    2      2      -0.0227    0.0235   0.0463   3903  2  FACTORY kers to regain control of the PLANT </S>
+    2      2      -0.0229    0.0041   0.0270   3787  2  FACTORY the workers ' entrance to the PLANT stadium , where rallies ar
+    2      2      -0.0232    0.0331   0.0563   3661  2  FACTORY oup of 67 women at the Hudson PLANT working in so-called clean
*    1      2      -0.0232    0.0671   0.0903   3972  1  LIVING   techniques for manufacturing PLANT protein products are large
+    2      2      -0.0234    0.0159   0.0393   3921  2  FACTORY  include accidents within the PLANT as well as accidents invol
+    2      2      -0.0240    0.0286   0.0525   3729  2  FACTORY escribed in the large body of PLANT shutdown literature , but
+    2      2      -0.0244    0.0232   0.0476   3866  2  FACTORY  , established a meat-packing PLANT to supply eastern markets
+    2      2      -0.0248    0.0145   0.0393   3660  2  FACTORY  engineers has scoured Newark PLANT records going back four de
+    2      2      -0.0259    0.0181   0.0440   3629  2  FACTORY ones in a vat of steel in the PLANT 's blastfurnace area </S>
+    2      2      -0.0259    0.0248   0.0507   3999  2  FACTORY s at its New Bedford , Mass , PLANT </S>
+    2      2      -0.0279    0.0124   0.0404   3893  2  FACTORY  in the sale of the Las Vegas PLANT and related assets </S>
+    2      2      -0.0293    0.0016   0.0310   3665  2  FACTORY r hardwood mills and a veneer PLANT </S>
+    2      2      -0.0294    0.0057   0.0351   3909  2  FACTORY ts car , the De Lorean , at a PLANT in Belfast , Northern Irel
+    2      2      -0.0297    0.0046   0.0344   3617  2  FACTORY t at GE 's Greenville , S.C , PLANT , with shipment set for mi
+    2      2      -0.0301    0.0013   0.0314   3818  2  FACTORY nd closed its 16-mm projector PLANT in 1985 </S>
+    2      2      -0.0302    0.0106   0.0408   3836  2  FACTORY              In a modern coal PLANT the combustion of one poun
+    2      2      -0.0306    0.0019   0.0325   3892  2  FACTORY  pride at the dedication of a PLANT expansion recently , and t
+    2      2      -0.0306    0.0098   0.0405   3890  2  FACTORY anks were being heated at the PLANT , when , in fact , managem
+    2      2      -0.0322    0.0125   0.0447   3615  2  FACTORY by occupations already in the PLANT and unions are interested
+    2      2      -0.0325    0.0337   0.0661   3844  2  FACTORY zation design of the chemical PLANT and traditional union stra
+    2      2      -0.0327    0.0503   0.0830   3985  2  FACTORY onstructing an aircraft parts PLANT in Macon , Ga </S>
+    2      2      -0.0331    0.0830   0.1161   3991  2  FACTORY oduction in 1989 and increase PLANT capacity 20 % </S>
+    2      2      -0.0334    0.0085   0.0419   3822  2  FACTORY s from its Camden offices and PLANT use its fitness center , d
+    2      2      -0.0340    0.0113   0.0453   3813  2  FACTORY he work force at its Evendale PLANT by 500 to 1,000 people </S
+    2      2      -0.0341    0.0326   0.0666   3742  2  FACTORY  parts currently built at the PLANT will be discontinued , but
+    2      2      -0.0357    0.0275   0.0632   3788  2  FACTORY  of a 1986 uranium-processing PLANT accident </S>
+    2      2      -0.0370    0.0239   0.0608   3702  2  FACTORY l administrator also said the PLANT 's problems " demonstrate
+    2      2      -0.0398    0.0139   0.0538   3618  2  FACTORY sodium-borohydride production PLANT in Delfzijl , The Netherla
+    2      2      -0.0402    0.0034   0.0436   3784  2  FACTORY  WHEELER CORP.'S cogeneration PLANT at Mt Carmel , Pa , will b
+    2      2      -0.0405    0.0454   0.0859   3798  2  FACTORY on 's influence on industrial PLANT location is currently unde
+    2      2      -0.0416    0.0047   0.0462   3863  2  FACTORY ker and the UAW locals at the PLANT broke off talks Friday , t
+    2      2      -0.0423    0.0313   0.0736   3620  2  FACTORY  and a natural gas processing PLANT near Calgary , Alberta , f
+    2      2      -0.0423    0.0122   0.0545   3932  2  FACTORY which doesn't link the Hudson PLANT problem with any particula
+    2      2      -0.0458    0.0048   0.0507   3819  2  FACTORY loans needed for the Monessen PLANT 's electric furnace </S>
+    2      2      -0.0460    0.0168   0.0628   3997  2  FACTORY aft , La , phosphate-chemical PLANT and its inventory from Fre
+    2      2      -0.0466    0.0162   0.0628   3775  2  FACTORY 985 that could have saved the PLANT , but that the union leade
+    2      2      -0.0473    0.0113   0.0586   3695  2  FACTORY rred costs of the Waterford 3 PLANT </S>
+    2      2      -0.0476    0.0113   0.0589   3621  2  FACTORY on of a liquefied-natural-gas PLANT , and Chase had hoped to r
+    2      2      -0.0489    0.0205   0.0693   3829  2  FACTORY ments made by supervisors and PLANT managers ; and paternalism
+    2      2      -0.0493    0.0217   0.0710   3638  2  FACTORY he program ; and at the third PLANT workers first supported th
+    2      2      -0.0496    0.0327   0.0823   3949  2  FACTORY ts and management of physical PLANT are presented </S>
+    2      2      -0.0496    0.0107   0.0603   3860  2  FACTORY               The Rancho Seco PLANT started up in 1974 , feedi
+    2      2      -0.0508    0.0067   0.0576   3920  2  FACTORY States in 1954 at the Detroit PLANT of the McLouth Steel Corpo
+    2      2      -0.0519    0.0233   0.0752   3923  2  FACTORY ent of India charged that the PLANT design was poor and that p
+    2      2      -0.0532    0.0089   0.0620   3739  2  FACTORY may still bid for work on the PLANT , they fear they won't be
+    2      2      -0.0533    0.0821   0.1354   3628  2  FACTORY ers and former workers at the PLANT -- and at a few other plan
+    2      2      -0.0534    0.0020   0.0553   3861  2  FACTORY ntract to operate a munitions PLANT at Shreveport , La </S>
+    2      2      -0.0534    0.0442   0.0976   3664  2  FACTORY rage hourly production rate , PLANT capacity and shipping meth
+    2      2      -0.0539    0.0114   0.0652   3811  2  FACTORY egarding costs connected with PLANT abandonments or rate disal
+    2      2      -0.0540    0.0414   0.0954   3889  2  FACTORY ford , N.Y , silicon-products PLANT </S>
+    2      2      -0.0546    0.0055   0.0601   3735  2  FACTORY er work from the Indianapolis PLANT </S>
+    2      2      -0.0575    0.0029   0.0605   3697  2  FACTORY  workers at the Hyundai Motor PLANT in Ulsan ; workers giving
+    2      2      -0.0579    0.0016   0.0596   3743  2  FACTORY er facilities at the Monessen PLANT </S>
+    2      2      -0.0598    0.0139   0.0738   3625  2  FACTORY cided to build a new chemical PLANT in Saukville , Wis , after
+    2      2      -0.0611    0.0018   0.0629   3703  2  FACTORY ible strike today at its Jeep PLANT in Toledo , Ohio , was unc
+    2      2      -0.0626    0.0052   0.0678   3934  2  FACTORY gency planning issues until a PLANT is substantially construct
+    2      2      -0.0632    0.0070   0.0702   3895  2  FACTORY rkers , who have occupied the PLANT since Jan. 14 in an effort
+    2      2      -0.0634    0.0202   0.0836   3705  2  FACTORY get cut for the Oklahoma City PLANT demanded by Firestone </S>
+    2      2      -0.0646    0.0045   0.0691   3941  2  FACTORY                           The PLANT also assembles , under a c
+    2      2      -0.0656    0.0065   0.0721   3865  2  FACTORY ls Corp unit , which owns the PLANT , paid the NRC a $310,000
+    2      2      -0.0660    0.0201   0.0861   3857  2  FACTORY Board that require disallowed PLANT costs to be treated as a r
+    2      2      -0.0670    0.0142   0.0812   3626  2  FACTORY y chemicals and salt said the PLANT is expected to be under co
+    2      2      -0.0683    0.0328   0.1011   3653  2  FACTORY vidual cost profiles for each PLANT and an industry supply sch
+    2      2      -0.0684    0.0051   0.0735   3806  2  FACTORY eptable representation of the PLANT 's operations </S>
+    2      2      -0.0687    0.0489   0.1176   3656  2  FACTORY number of production cuts and PLANT closures industrywide in r
+    2      2      -0.0708    0.0172   0.0880   3669  2  FACTORY in its search for a new steel PLANT site </S>
+    2      2      -0.0736    0.0030   0.0766   3631  2  FACTORY to 1,000 jobs at a jet-engine PLANT at Evendale , Ohio , over
+    2      2      -0.0768    0.0051   0.0818   3781  2  FACTORY act for operating a munitions PLANT at Independence , Mo </S>
+    2      2      -0.0782    0.0033   0.0816   3774  2  FACTORY d a new liquids manufacturing PLANT in Cambridge , Ohio </S>
+    2      2      -0.0787    0.0108   0.0894   3700  2  FACTORY t its Indianapolis electrical PLANT </S>
+    2      2      -0.0793    0.0271   0.1064   3931  2  FACTORY  and one-time expenses from a PLANT closure and other cost-cut
+    2      2      -0.0809    0.0132   0.0940   3672  2  FACTORY  example , the Racine tractor PLANT would operate with fewer t
+    2      2      -0.0832    0.0102   0.0934   3855  2  FACTORY  its California joint-venture PLANT with Toyota Motor Corp bec
+    2      2      -0.0854    0.0167   0.1021   3704  2  FACTORY distinction between permanent PLANT closings and short-term ,
+    2      2      -0.0870    0.0459   0.1329   3936  2  FACTORY ll begin in early 1988 on the PLANT , to have production capac
+    2      2      -0.0882    0.0038   0.0920   3786  2  FACTORY  Firestone tire manufacturing PLANT in Salinas , Calif </S>
+    2      2      -0.0901    0.0385   0.1285   3662  2  FACTORY            The gas processing PLANT , which was recently compl
+    2      2      -0.0916    0.0158   0.1074   3791  2  FACTORY rs in a garment manufacturing PLANT </S>
+    2      2      -0.0919    0.0234   0.1153   3693  2  FACTORY  to make full use of existing PLANT and equipment </S>
+    2      2      -0.0927    0.0170   0.1097   3658  2  FACTORY an incident in 1985 -- when a PLANT operator mistakenly starte
+    2      2      -0.0941    0.0240   0.1181   3616  2  FACTORY ike houses , collectibles and PLANT equipment " </S>
+    2      2      -0.0990    0.0047   0.1036   3666  2  FACTORY tor Corp said it will build a PLANT in Youngstown , Ohio , to
+    2      2      -0.0992    0.0014   0.1006   3619  2  FACTORY loney " the argument that the PLANT was operating , Mr Hall sa
+    2      2      -0.1001    0.0244   0.1245   3862  2  FACTORY ving the railroad 's physical PLANT and making severance payme
+    2      2      -0.1005    0.0082   0.1087   3901  2  FACTORY ure will build a $100 million PLANT in Bethlehem , Pa , to con
+    2      2      -0.1037    0.0676   0.1713   3741  2  FACTORY too early to determine if the PLANT closing will result in a c
+    2      2      -0.1038    0.0047   0.1085   3668  2  FACTORY          Unit I of the Vogtle PLANT currently is operating at
+    2      2      -0.1053    0.0096   0.1149   3632  2  FACTORY  as two years ago to have the PLANT in commercial operation by
+    2      2      -0.1066    0.0606   0.1672   3780  2  FACTORY ditions , and for a completed PLANT if certain costs for the p
+    2      2      -0.1083    0.0229   0.1311   3902  2  FACTORY        The company 's largest PLANT investment in at least 10
+    2      2      -0.1100    0.0100   0.1200   3935  2  FACTORY ion of capacity at its engine PLANT in Anna , Ohio and said it
+    2      2      -0.1110    0.0083   0.1193   3894  2  FACTORY nt in the Millstone 3 nuclear PLANT in Waterford , Conn , that
+    2      2      -0.1114    0.0150   0.1264   3900  2  FACTORY r power plant to a coal-fired PLANT is scheduled for completio
+    2      2      -0.1118    0.0138   0.1256   3930  2  FACTORY already started to design the PLANT and that construction will
+    2      2      -0.1132    0.0507   0.1639   3768  2  FACTORY ibution to the study of power PLANT siting as a political prob
+    2      2      -0.1167    0.0319   0.1485   3927  2  FACTORY of design , package filling , PLANT operation , and other serv
+    2      2      -0.1201    0.0027   0.1228   3624  2  FACTORY . 's Alvin W Vogtle I nuclear PLANT in Waynesboro , Ga </S>
+    2      2      -0.1222    0.0274   0.1496   3779  2  FACTORY l have to make , assuming the PLANT starts operation early nex
+    2      2      -0.1236    0.0191   0.1427   3670  2  FACTORY perating its own cogeneration PLANT , capable of producing 80
+    2      2      -0.1241    0.0173   0.1413   3782  2  FACTORY ion 's Chernobyl atomic-power PLANT , which cut farm income ;
+    2      2      -0.1248    0.0064   0.1312   3690  2  FACTORY       The first nuclear power PLANT was placed aboard the subm
+    2      2      -0.1273    0.0531   0.1804   3810  2  FACTORY ction for workers affected by PLANT closings </S>
+    2      2      -0.1274    0.0441   0.1715   3816  2  FACTORY t to increase the cost of the PLANT , which is to be completed
+    2      2      -0.1319    0.0024   0.1344   3887  2  FACTORY ontract for an electric power PLANT in Mendota , Calif </S>
+    2      2      -0.1341    0.0103   0.1444   3633  2  FACTORY 8 billion Waterford 3 nuclear PLANT north of New Orleans </S>
+    2      2      -0.1375    0.0168   0.1544   3882  2  FACTORY the Heber demonstration power PLANT in California , which is s
+    2      2      -0.1450    0.0300   0.1750   3896  2  FACTORY  large coal-fueled generating PLANT that went into operation l
+    2      2      -0.1466    0.0159   0.1626   3778  2  FACTORY ing the impact of the nuclear PLANT delay and said it expects
+    2      2      -0.1515    0.0471   0.1986   3820  2  FACTORY                    Moreover , PLANT closings would increase th
+    2      2      -0.1533    0.0068   0.1601   3864  2  FACTORY  workers dismissed because of PLANT closings , it sets up a $4
+    2      2      -0.1546    0.0120   0.1665   3663  2  FACTORY upled with recently announced PLANT closings , will have " onl
+    2      2      -0.1563    0.0085   0.1648   3776  2  FACTORY  Campbell 's electrical power PLANT , says he used to drink a
+    2      2      -0.1586    0.0078   0.1664   3636  2  FACTORY t the Three Mile Island power PLANT near Harrisburg brought in
+    2      2      -0.1595    0.0104   0.1699   3701  2  FACTORY its Oklahoma City tire-making PLANT </S>
+    2      2      -0.1667    0.0036   0.1703   3907  2  FACTORY s , and a hydroelectric power PLANT has been constructed at th
+    2      2      -0.1688    0.0164   0.1852   3929  2  FACTORY                           The PLANT will be closed by the end
+    2      2      -0.1697    0.0467   0.2163   3891  2  FACTORY ing Heights , Mich , assembly PLANT for the week of April 20 <
+    2      2      -0.1698    0.0212   0.1910   3821  2  FACTORY dy on a superconducting power PLANT and plans to have a workin
+    2      2      -0.1741    0.0059   0.1800   3898  2  FACTORY uy all surplus power from the PLANT </S>
+    2      2      -0.1778    0.0359   0.2137   3899  2  FACTORY  refuses to accept this power PLANT " </S>
+    2      2      -0.1787    0.0082   0.1870   3939  2  FACTORY  to a proposed electric power PLANT in Texas </S>
+    2      2      -0.1807    0.0358   0.2165   3859  2  FACTORY  of $8.6 million related to a PLANT closing </S>
+    2      2      -0.1813    0.0122   0.1935   3659  2  FACTORY r Co , project manager of the PLANT , said the company has to
+    2      2      -0.1972    0.0166   0.2138   3897  2  FACTORY canceled Zimmer nuclear power PLANT to a coal-fired plant is s
+    2      2      -0.1984    0.0228   0.2211   3938  2  FACTORY cord a domestic nuclear power PLANT in its second-quarter sale
+    2      2      -0.1990    0.0142   0.2132   3856  2  FACTORY its investment in two nuclear PLANT units </S>
+    2      2      -0.2055    0.0072   0.2128   3740  2  FACTORY ts Belvidere , Ill , assembly PLANT </S>
+    2      2      -0.2107    0.0145   0.2253   3734  2  FACTORY maker said it would close the PLANT , which makes passenger-ca
+    2      2      -0.2126    0.0104   0.2230   3933  2  FACTORY  , which includes an assembly PLANT and engine and stamping op
+    2      2      -0.2191    0.0220   0.2411   3737  2  FACTORY ties to write off any nuclear PLANT costs they can't recover w
+    2      2      -0.2271    0.0109   0.2380   3785  2  FACTORY to the Seabrook nuclear power PLANT just two miles north of Ma
+    2      2      -0.2284    0.0338   0.2622   3888  2  FACTORY y close its only U.S assembly PLANT for the first time in more
+    2      2      -0.2319    0.0161   0.2481   3937  2  FACTORY ndoned portion of the nuclear PLANT </S>
+    2      2      -0.2753    0.0119   0.2873   3886  2  FACTORY ent would drive nuclear power PLANT operators to look abroad f
+    2      2      -0.3358    0.0112   0.3470   3627  2  FACTORY to the Seabrook nuclear power PLANT are about $2.5 million </S
================================================================================================================================

plant stemmed Prediction Accuracy: 0.9175


Evaluation

At the end I just call all the implemented functions from part1-2 for different files and models.
An example of the output table is shown bellow. 

===================================================================================================
   Stemming    |  Position Weighting  |  Local Collocation Modelling  |        Accuracy
               |                      |                               | tank | plant | pers/place
               
===================================================================================================
1  unstemmed   |  #0-uniform          |  #1-bag-of-words              | 0.93 | 0.92  |    0.77
2  stemmed     |  #1-expndecay        |  #1-bag-of-words              | 0.94 | 0.92  |    0.81
3  unstemmed   |  #1-expndecay        |  #1-bag-of-words              | 0.94 | 0.94  |    0.78
4  unstemmed   |  #1-expndecay        |  #2-adjacent-separate-LR      | 0.91 | 0.92  |    0.78
5  unstemmed   |  #2-stepped          |  #1-bag-of-words              | 0.94 | 0.94  |    0.78
6  unstemmed   |  #3-yours            |  #1-bag-of-words              | 0.91 | 0.89  |    0.78


Extensions to the Classification Model

I implemented three further classification models of Naive Bayesian, K nearest neighbours and hierarchal clustering using sklearn.
To be able to fit the dictionaries into the sklearn models, I first got all the unique words for the entire doc_vectors.
After doing this I set each vector to have the same number of elements as unique words. The position for each word is the same
across all vectors. When a token shows up for every document I put the weight according to it's position on the unique word index.
After that I split the entire data set to 77% train and 33% test. It seems that running a model with this many features locally
creates a memory error but it works in the Ugrad servers. I do this process for both Naive Bayesian and K nearest neighbours
classifiers by calling the functions from sklearn. For clustering, I do the same process to create the data_set from the doc_vectors
but from the lables returned from the function fit_predict() I compare it to the original sensenum lables. Since we don't know
which cluster belongs to which label, I take the highest average of both probable accuracies since that makes the most sense.
An example of the results for the extension classifiers are shown bellow.

Running this part takes approximately 5 min or longer.

=================================================================================================================
   Model       |  Stemming    |  Position Weighting  |  Local Collocation Modelling  |        Accuracy
               |              |                      |                               | tank | plant | pers/place
               
=================================================================================================================
1  Bayesian    |  unstemmed   |  #0-uniform          |  #1-bag-of-words              | 0.90 | 0.96  |    0.66
2  KNN         |  unstemmed   |  #1-expndecay        |  #1-bag-of-words              | 0.81 | 0.77  |    0.71
3  Clustering  |  unstemmed   |  #1-expndecay        |  #1-bag-of-words              | 0.58 | 0.53  |    0.54

