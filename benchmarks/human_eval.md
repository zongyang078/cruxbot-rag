# Relevance grading sheet

79 (query, passage) pairs sampled across the benchmark, stratified
by the judge's grade. **The judge's grades are not shown** — they are held in
`human_eval_key.json` and compared only after this sheet is filled in.

Each item is one retrieved passage, quoted verbatim from the corpus. **These are
not generated answers.** No model wrote them and nothing was summarised or
tidied up; most are forum posts, accident reports, or route descriptions. Grade
the information, not the prose — a blunt one-line forum reply that answers the
question beats a well-written article that does not.

The same query appears several times, paired with a different passage each time
(35 queries across these 79 pairs). Judge each pair on its own; do not rank a
passage against the others for its query.

Sources are withheld deliberately. Knowing a passage came from an AAC accident
report rather than a forum thread would anchor the grade on authority instead of
on whether it answers the question.

Fill in each `GRADE:` line with 0, 1, 2, or 3:

- **3** — Directly answers the question. A climber would need nothing else.
- **2** — Useful but partial. Addresses the question without fully answering it.
- **1** — On topic but does not address the question. Background at best.
- **0** — Irrelevant to the question.

Ask one thing only: if a climber read this passage and nothing else, how much of
*that* question is answered? A route description is not relevant to a training
question merely because both concern climbing.

---

### G07|4b47d124ee49_c0

**Q1. Best chalk bag for bouldering?**

> Topic: Best chalk bag for large hands?. Just strap a chalk bucket to ya self.

GRADE: 2

### S03|78fe5e3506fc_c0

**Q2. What causes rappelling accidents in climbing?**

> Rappel Error Location: Utah, Wasatch Range, Big Cottonwood Canyon, Storm Mountain Type: Accident Reports Author: Salt Lake County Search and Rescue On July 7, a male climber in his 20s fell to the ground while rappelling from a multi-pitch sport route called Addis Ababa (two pitches, 5.7). The climber and two partners had finished the route, and two of the three climbers had completed the rappels when they saw the third fall from near the first-pitch anchors to the ground. The patient had open fractures in both legs and a head injury, among other injuries. Rescuers lowered him down low-angle terrain for about 100 feet, and then he was carried to Life Flight for a trip to a local hospital. (Source: Salt Lake County Search and Rescue.) ANALYSIS It s not clear what caused the fall, but rescuers believe the climber loaded his ATC-style rappel device incorrectly. Weight-testing the rappel set

GRADE: 3

### S03|b600356dda13_c0

**Q3. What causes rappelling accidents in climbing?**

> Topic: What that rope do?. Since you're in WA If you plan to climb much at index get a 70m. It's the standard for the area and there have been several accidents from using too short of a rope when lowering or rappelling.

GRADE: 2

### G05|8e9c467f7caa_c0

**Q4. Review of Black Diamond ATC Guide belay device**

> Topic: Belaying with an autolock off the anchor. Black Diamond has a new device called the ATC Guide. It functions pretty much identical to the Petzl Reverso but it has a couple of extra features. A small "release ring" is slung with some cord. A tug on this is usually enough to release the autoblock. You can also run this release sling through the anchors and to your harness to use your body weight to release the autoblock. This has the added advantage of allowing both your hands to be free although you must never release your brake hand. The ATC Guide also has two friction modes for normal belaying or rapping. It is 2 cheaper than the Reverso. The Reverso has a little more room to maneuver ropes and biners. Weight is identical. The Guide is compatible with ropes from 7.7 to 11 mm.

GRADE: 3

### G01|0cc9d63bd64a_c0

**Q5. What is the best belay device for multi-pitch climbing?**

> Topic: Guide Mode usage. Pivot is now my standard device for long multi-pitches. For short multi-pitch I also carry a GriGri for its ease of use in top belaying.

GRADE: 3

### G04|01f931e1ff2a_c0

**Q6. Best harness for beginners?**

> Topic: Best climbing gear for beginners. For a beginner gym climber you really want shoes harness a belay device and a locking carabiner - that's about it (and maybe a chalk bag). For shoes it's all about fit - there are a number of good brands (evolv la spprtiva butora 5.10 scarpa) but it all depends on your foot shape. Find a store where you can try many and talk to the people there to help you find a good fit. Belay device - you have two main type - tube devices (BD ATC) or assisted devices (grigri). There are pros and cons to each and many people would recommend a grigri for beginners. The BD momentum harness is petty decent relatively cheap option for beginners. In addition to BD other brands are petzl arxteryx mammut. Edit: climb first with gym rental equipment for a bit to gain more experience and decide whether you want to stick with it

GRADE: 2

### R02|2ff27b2afadf_c0

**Q7. What are some good beginner trad routes in Red River Gorge?**

> Funhouse is a 5.7 Trad climbing route. located in Wall of Denial Northern Gorge Red River Gorge Kentucky. rated 3.0 stars. 1 pitches. 80.0 feet long. This route just keeps going. The climbing is varied along the entire route. The crack is wide then narrower then jumbled. There is some face some crack some weirdness. The rock quality also varies a little and the upper section of the route has some potentially loose rock. The route is in a great setting with the sound of the whitewater of the Upper Red River Gorge echoing from the opposite walls of the gorge as you climb. The belay ledge is a neat little perch and seems about as far from civilization as you can get in this part of Kentucky.. Protection: Standard rack maybe a #4 camalot as well. There are bolted anchors.

GRADE: 3

### T08|692d9c4c437b_c0

**Q8. What stretches help with hip flexibility for climbing?**

> Topic: Climbing pants for short guy/girl. Booty shorts: the elasticity will work wonders.

GRADE: 0

### G06|fa78656afd70_c0

**Q9. What helmet should I use for outdoor climbing?**

> Topic: Helmets in climbing. Wear em always from the moment I drop my pack to the hike out. Pros should wear em. yarp.

GRADE: 1

### S09|0e9945f58a09_c0

**Q10. How dangerous is free soloing?**

> Topic: The "Free Solo" Effect. How is this a gear discussion?

GRADE: 0

### G10|43da242869c3_c1

**Q11. Are assisted braking belay devices safer than tube style?**

> y interest is in assisted braking. I want a device that allows for rope management at least as good preferably better than an ATC while supplying a lot more braking when a really big impact occurs. The only real argument for genuine locking devices is the idea that the belayer could be knocked out. This can happen but it is rare and I view it as one of the intrinsic risks of our activity. Thanks again for all the valuable information you've shared!

GRADE: 2

### T05|6a5ff78128b5_c0

**Q12. How to prevent climbing injuries in fingers and elbows?**

> Topic: Training boards?. Based on my own experiences and opinion I'll offer the following. First "just climb" is probably sound advice for most of us. Especially if we are talking about climbing at the sub-elite grades. However a bit of general fitness that includes antagonist muscles and maybe some cardio may help. For example push ups pull ups and cardio of choice once or twice a week in addition to your climbing once or twice a week is probably sufficient. Just make sure you leave enough time to recover between climbs and workouts. Also be aware of the risk for overuse injuries. Especially if you decide to train on a board. Watch out for the dreaded Tennis Elbow and/or Golfer's Elbow. They start with a stubborn pain around your inner or outer elbows respectively. If you are affected by either of these issues and you ignore it it is a long long long miserable and frustrating recovery. 

GRADE: 3

### S02|b58af88b00be_c0

**Q13. How do I safely clean a sport climbing anchor?**

> Topic: Cleaning climbing holds. What's the best/easiest way to clean holds?

GRADE: 1

### S03|9098ae02f3dc_c3

**Q14. What causes rappelling accidents in climbing?**

> 22 4 Unknown 25 1 Month of Year 1952 1953 January 0 1 February 0 1 March 0 0 April 3 0 May 3 1 June 0 0 July 11 6 August 11 7 September 3 4 October 3 2 November 1 1 December 0 0 ANALYSIS OF ACCIDENTS As in the past two years the various accidents reported have been analyzed and it is noteworthy that the causes seem to be following a regular pattern with only slight fluctuations from year to year. This year, however, two causes stand out and deserve more careful attention. The first of these is apparent. It is rappelling. In 1953 there were three accidents associated with failure en rappel, whereas there had been only five in the previous six years. This may represent only a random variation but it still focuses our attention on this procedure which should not be so hazardous. In previous years accidents have occurred as the rappel was being established. This type of accident is caused me

GRADE: 2

### S02|0e19fb14ddad_c0

**Q15. How do I safely clean a sport climbing anchor?**

> Topic: PAS Advice. It really doesn't matter. The most important part of sport climbing anchor cleaning is that you NEVER GO OFF BELAY.

GRADE: 3

### R05|1e664ca9a57e_c0

**Q16. What are some classic 5.12 routes in Smith Rock Oregon?**

> Resuscitation is a 5.12c Sport climbing route. located in (2) Wildfire Wall West Side (z) Lower Gorge Smith Rock Central Oregon Oregon. rated 3.4 stars. 1 pitches. The sport routes in the lower gorge can be thoughtful delicate and powerful! This route is a culmination of the 5.12 routes and is nothing short of fantastic. It's bad-ass with the crux above the fourth bolt - stick the move or re-suss your beta.. Protection: 7 bolts

GRADE: 3

### S10|130402a2eae2_c2

**Q17. What should I do if my climbing partner falls and is injured?**

> d time. However, sometime during his fall he either caught his crampon on a rock or his crampon impacted the wall, causing his leg to snap. The victim called down to his partner and told him that he had broken his leg and could not climb down. The partner immediately made a 911 cell phone call for help and initiated a SPOT beacon alert before he began to assist the victim down to the belay ledge. Both men had been climbing three to four years and considered themselves to be competent on 5.8 5.9 traditional routes, with some mixed climbing experience. They were properly equipped to deal with a variety of conditions (snow, ice, and rock). Unfortunately, on this particular day, the rock on the route was covered with a light dusting of unconsolidated snow and no ice not conditions for which they were suited. Rangers had conveyed the conditions on the route to the climbers the day before they

GRADE: 3

### S08|510d49c3bf2c_c3

**Q18. Fatal climbing accidents caused by equipment failure**

> plus age group which could be interpreted to support this). An explanation of the relatively constant ratio with increasing age is undoubtedly complex, however, it does seem to be evident that fatal accidents in the youngest and less experienced account for a large percentage of the mountaineering accidents. Figure 6 presents the fatal and non-fatal accidents according to apparent primary cause. As in Figure 3, a slip or fall on rock carries a much higher chance of a fatal outcome than a similar event on snow. Being struck by a falling object or an accident occurring during rappelling were the next two most common causes. Neither appeared to carry as great a chance of a fatality as a slip or fall on rock. The relatively high number of accidents due to falling objects emphasizes the importance of wearing protective headgear. There appears to be much variation in the protection offered by 

GRADE: 0

### G04|ce92f7daf0c4_c0

**Q19. Best harness for beginners?**

> Topic: Good Trad harness??. BD Momentum (not Momentum AL) is a good beginner all-purpose harness. 4 gear loops adjustable legs etc. I'm on my second one now and use it for everything from trad to alpine ice.

GRADE: 2

### R06|3af966e863a4_c0

**Q20. Find me a 5.8 sport route in Joshua Tree**

> Unknown 5.8 #2 is a 5.8 sport climbing route. located in California > Joshua Tree National Park > Pinto Basin > OZ Area > Magic Mountain Area > Magic Mountain

GRADE: 3

### G05|29822a852b4a_c0

**Q21. Review of Black Diamond ATC Guide belay device**

> Topic: Beginner In Need of Gear Advice. Any new harness will do the job they are all about equal in terms of strength and reliability. As you gain experience and income you can start getting picky about things like padding gear loops etc. The Black Diamond Alpine Bod is as cheap as they get and works just fine. rei.com/product/699550/blac Carabiners: auto locking carabiners are nice but screwgates are cheaper and do the job just as well. The better quality/more expensive locking carabiners tend to function more smoothly both in opening and closing the gate and locking/unlocking the biner. Belay devices: for a long time the Black Diamond ATC belay device was the standard and they still work quite well. The ATC Guide has a few advantages for multi-pitch climbing or certain toprope applications if you want to spend a little more. Shoes: The higher priced shoes can be resoled a few more time

GRADE: 2

### S07|545c536401b6_c0

**Q22. What are common belaying mistakes that lead to accidents?**

> Topic: Your ATC is unsafe. True RGold. I guess I've been lucky. In my entire time climbing I have never had a partner that I have belayed that has gotten injured save for a few bumps and scrapes while hip belaying or using an ATC. I have gotten injured a few times while being belayed . but it was not the belay devices fault. It was the belayer. Case in point; I was 2 feet above my last piece and fell 50 feet. No. My last piece did not pull out. I think more serious accidents with any kind of belay device are caused by inattentive belayers rather than the devices.

GRADE: 1

### T08|d0e0df6cf82b_c0

**Q23. What stretches help with hip flexibility for climbing?**

> Topic: Rope length. 70 m 9.8 +/- 0.3 ish mm with a dry coat. Provides flexibility for rappels and safety on longer routes where lowering your partner off the end of your rope is less than ideal. If you live in Kansas and the biggest cliff you plan to climb 8 feet tall disregard what I said.

GRADE: 0

### S10|4e18efa0869d_c2

**Q24. What should I do if my climbing partner falls and is injured?**

> nt. I did not want to be under that roof if my partner fell, so I decided to stand farther left. Unfortunately, this meant there was a lot of extra rope out and I was pulled up and into a swing by the leader s fall. I still feel it was the right decision to stand further left because I might have broken my neck by hitting the roof instead. A ground anchor might have prevented the dangerous swing. There were no good opportunities to build an anchor in the rock at the base, though a small tree nearby might have withstood the force of the swing. The other option would have been to scramble up to the ledge and tie into the intermediate anchor at the rappel station in order to belay the leader. My climbing partner is 45 pounds heavier than I am, which is right on the limit of what is considered safe. To compensate for this, we could have used the Edelrid Ohm, which is clipped to the first bol

GRADE: 0

### R08|64d71c25e298_c0

**Q25. Suggest a beginner-friendly climbing route in Utah**

> Topic: Rats and Gear storage. I'd suggest trying one of these rat bashing sticks

GRADE: 0

### T10|742210c880a4_c0

**Q26. Should I campus board as a beginner?**

> Topic: hang board options. Rock Prodigy Training Center

GRADE: 1

### T01|2f2cfaa202d3_c0

**Q27. How should I train finger strength for climbing?**

> Topic: Training with hangboards. Check out the training guide on Metolius's website. I'd suggest getting one for your home. Then follow the guide. Took me a few months to work up to the intermediate level workout but really helped finger/grip strength.

GRADE: 3

### T10|0ea07049e172_c0

**Q28. Should I campus board as a beginner?**

> Topic: Professional Campus Board. I'm sure your climbing community could build a great wall and install pre made rungs. Just my suggestion

GRADE: 1

### R04|35a61a2a1a6b_c4

**Q29. Recommend a multi-pitch route in Yosemite for intermediate climbers**

> is great alpine route. To pick the most classic routes in Yosemite Valley must have been difficult. On the east buttress of Middle Cathedral Rock, the authors stress it is the quality of climbing that recommends it. With this I can agree, for it is a magnificent one-day route on superb rock. The ethic of overbolting is condemned on Salathé Wall, where a proliferation of bolts now marks the route. For those who wish to escape from the throngs of climbers in Yosemite, and the likelihood of having to share the serenity of a route with others, there is always the High Sierra. Or is there even solitude here any longer? A fine climb suddenly becomes popular; Charlotte Dome (which may now become even more popular because of inclusion in this book) recently witnessed the arrival of two separate climbing teams. Where to go for seclusion? Fred Beckey

GRADE: 1

### T09|c24d7fd1d319_c2

**Q30. How to overcome a climbing plateau at 5.11?**

> ry little exposure to other than friends mentioning their past or current struggles but only ever tentatively and with no indication for speaking further on it but I'd like to know more so I can sympathise more and be a better friend and supporter for anyone around me going through these struggles. I can appreciate that it's really a case by case thing and everyone goes through the struggle of whatever kind of ED they have in probably very different ways but if there's like some resources that you would stand by in regards to educating myself, or like some resources to jump off from that are better than a cursory google I'd welcome them. xxxx Honestly the way I see it anything you do to lose weight will benefit your climbing pretty significantly. Climbing more than once a week would probably help, but any kind of regular exercise will also be good. Maybe a mix of climbing, basic strength

GRADE: 0

### R06|edff3a93c57e_c0

**Q31. Find me a 5.8 sport route in Joshua Tree**

> Crack 5 is a 5.9 Trad climbing route. located in Isles Corridor - Left Side Isles Corridor Isles in the Sky Split Rocks Joshua Tree National Park California. rated 2.8 stars. 1 pitches. 50.0 feet long. Great splitter route! Rated 5.9+ in the Vogel Joshua Tree guide but so straight forward that 5.9 is more apt if you've wired your handjamming technique. Towards the top the crack forks: continue up the left side for more of the same handjamming (5.9) or go right for a steep fingers section (5.10a/b). Take the right fork for an exciting pumpy finish (highly recommended)!. Protection: Lots and lots of hand to wide-hand size gear. If you plan on finishing with the fingers section take a couple finger sizes (I placed two yellow Metolius here). Nuts may be useful for the finger section as well.

GRADE: 2

### S10|ceb133a1e880_c1

**Q32. What should I do if my climbing partner falls and is injured?**

> ow-angle terrain (often found on easy climbs in the 5.6 5.8 range), a leader fall of almost any length can result in hitting a ledge, causing injury. Additionally, the risk of a head or neck injury is greatly increased in low-angle terrain, where a backward fall could flip the climber upside down before hitting the rock. This climber was wearing a helmet. Climbing with a partner is usually safer. Having another uninjured person to help in the case of an accident could be the difference between being able to self-rescue or needing to call search and rescue. In this case, luckily, the climber had cell phone service. Be sure to know when and where you may have reception in case of an emergency. (Source: Yosemite National Park Climbing Rangers.)

GRADE: 1

### S07|ccea9bee0b97_c0

**Q33. What are common belaying mistakes that lead to accidents?**

> Preventing Lowering Accidents Type: Editorials And Prefaces Author: Dougald MacDonald I WOULD BE VERY HAPPY if every reader of this book would make a simple three-step pledge. Doing so might save a few lives. A few of your own lives. This year s edition reports a worrying leap in the number of accidents while lowering or preparing to lower from anchors atop single-pitch climbs. Having seen growing numbers of such accidents in recent years, we introduced lowering errors as a primary accident cause in our data tables in the 2016 edition; the errors include too-short ropes slipping through a belayer s device, communication mix-ups, and failure to retie properly at an anchor. In 2016, we recorded five such incidents. The following year, we counted six. This year we documented 12 lowering accidents. Now this could be just a statistical blip. I sure hope so. It also might reflect the much-disc

GRADE: 2

### R08|43201fc037e8_c0

**Q34. Suggest a beginner-friendly climbing route in Utah**

> Local Motion is a 5.7+ PG13 Sport, TR climbing route. located in The Monolith Archangel Valley Sport and Traditional Climbing Hatcher Pass Anchorage & South Central Alaska Alaska. rated 2.4 stars. 1 pitches. 150.0 feet long. This route isn't as easy as it looks nor as easy as the old hand-drawn guides say it is and the runout makes it not much of a beginner-friendly lead. The crux is on a bulge near the top. Walk off or rap with two ropes.. Protection: 6 bolts and an optional red alien. Two bolt anchor.

GRADE: 2

### S01|29790411bb85_c2

**Q35. What are the most common lead climbing accidents?**

> a 134 14 4 1 Frostbite 99 9 7 0 Dislocation 91 11 4 1 Puncture 37 11 2 0 Acute Mountain Sickness 36 0 1 0 HAPE 62 0 1 0 HACE 20 0 1 0 Other5 240 37 18 6 None 165 176 11 3 N.B. Some accidents happen when climbers are at the top or bottom of a route, not climbing. They may be setting up a belay or rappel or are just not anchored when they fall. (This category was created in 2001 to replace unknown. ) 1These illnesses/injuries, which led directly or indirectly to the accident, included: exhaustion (7), dehydration (4), fatigue (2), syncope, HAPE, HACE, pulmonary infection. 2These include an inadequate knot (2), rope too short (2), and improper use of descending device, and no experience. 3This category was set up originally for ski mountaineering. Backcountry touring or snowshoeing incidents even if one gets avalanched are not in the data. 4These include: stranded because of dropping climbi

GRADE: 2

### R10|5d4e3b1fdb07_c0

**Q36. Recommend a V0-V2 bouldering area near San Francisco**

> V0 is a V0 bouldering climbing route. located in California > San Francisco Bay Area > Wine Country > Salt Point State Park > Playground, The > Playground Boulders > Blockhead Boulder

GRADE: 2

### S01|60cc38b082c6_c0

**Q37. What are the most common lead climbing accidents?**

> Know The Ropes: Protection Location: The "Ins And Outs" Of Sport And Trad Climbing Protection Type: Feature Article Author: Ron Funderburke & Karsten Delap / Photos: Dougald MacDonald & Erik Rieger Along with a rope, protection is the most essential part of the climbing system. A bolt and quickdraw, a cam or nut these are the things that keep climbers from taking dangerous ledge falls or hitting the ground. While not the most common cause of incidents reported in Accidents , failures of a lead climber s protection system occur frequently. In 2012, for example, Accidents recorded data on 11 incidents where protection pulling out was the immediate cause of an accident. Placing no protection or inadequate protection were contributory causes for 27 accidents. Similar numbers were reported in 2013. So the lead climber s protection system, or lack thereof, is clearly worthy of consideration as

GRADE: 2

### R06|3d963134673f_c0

**Q38. Find me a 5.8 sport route in Joshua Tree**

> Medicinal Marijuana is a 5.8 sport climbing route. located in California > Joshua Tree National Park > Central Joshua Tree > Sheep Pass Area > Oyster Bar Area > Conrad Rock > Conrad Rock - (W. Face)

GRADE: 3

### R06|0fceea07ec2f_c0

**Q39. Find me a 5.8 sport route in Joshua Tree**

> The Goop Gobbler is a 5.8 sport climbing route. located in California > Joshua Tree National Park > Pinto Basin > Indian Head Area > Safe Sex Zone

GRADE: 3

### S05|4feb7384249a_c0

**Q40. What happened in climbing accidents at Red River Gorge?**

> Fall on Rock, Inadequate Belay, Distraction Location: Kentucky, Red River Gorge, Military Wall Type: Accident Reports A large group was gathered at Military Wall on September 12 playing music, possibly loud enough to make communication between climber and belayer difficult. Climber was getting into the upper knee bar on Reliquary (5.12b) when he fell near the last bolt. The climber landed on the belayer s dog, killing it instantly. The belayer was holding the GriGri in her right hand with fingers over the cam, preventing it from locking, and said she didn t know how it could have happened. Worse yet, she said this was the third time it has happened to her. Analysis Crags are becoming more crowded and distractions occur. Music, dogs, kids, cats, bears, etc. Pay attention to the climber! That person s life is in your hands! And don t pick up a belay device if you don t have proper training

GRADE: 3

### R09|54fa6c29d357_c0

**Q41. What is the hardest sport route in El Potrero Chico?**

> Topic: Another Five-Ten thread. What is the correct 5.10 shoe for multi-pitch 5.12 climbing in El Potrero Chico? Tech limestone (I'd imagine). My quiver: Anasazi VCS Galileo Pinks Blancos Many thanks.

GRADE: 1

### R06|050ee8a9cade_c0

**Q42. Find me a 5.8 sport route in Joshua Tree**

> Poker in the Rear is a 5.8 sport climbing route. located in California > Joshua Tree National Park > Central Joshua Tree > Echo Rock Area > Big Hunk > Poker Face. First ascent: Kelly Vaught and Frank Bentwood, 2015

GRADE: 3

### S04|0520aed7e5b4_c1

**Q43. How to avoid rockfall injuries while climbing outdoors?**

> d they very possibly could have dislodged a rock above the climbing cliff. She notes, Rocks triggered by bighorns are a potential hazard to climbers, though it s extremely rare as the sheep are very shy and tend to avoid people. Nonetheless, she said, I personally have seen falling rock caused by wildlife. For the same reasons people like to climb in an area, it s also good for a bighorn habitat. It s easy to be lulled into a sense of security at a very popular crag. The moderate grades, easy access, and sun-drenched aspect make this particular cliff a busy year-round destination. However, this is not a gym, and natural rockfall should be expected at any crag in a mountainous or canyon environment. (This is especially true after heavy rain or snow or during wind storms, all of which can dislodge rocks.) Adopting an alpinist s sense of mountain awareness can help prevent such accidents. W

GRADE: 1

### T09|df054c410b3f_c0

**Q44. How to overcome a climbing plateau at 5.11?**

> Sunday Roast is a 5.11+ sport, alpine climbing route. located in Utah > South Central Utah > Jungle, on the Aquarius Plateau > Shangri La. First ascent: Andy Ross with John & Susan from Michigan

GRADE: 0

### T04|305f505cffad_c2

**Q45. Tips for climbing slab routes?**

> k until a horizontal crack appears on the left (takes a great #2!) and then traverse back into the chimney. Continue up until there are some large wedged chalkstones that make for a great belay spot. This pitch can be a bit intimidating as the rock can get rather "kitty littery" if you get even a little off route. Hold on tight. Pitch 4 - 5.6/7 Continue up the large crack for 70ish feet. Stop here to belay or if you are feeling brave and don't mind the rope drag continue to the summit. Build an anchor on the ridgeline of the formation. Pitch 5 - 5.3/5.4 Get your slab runout face on! Really easy climbing with one placement in the 100 ft of climbing. Heels down and mantle your way on up until you reach the anchor bolts at the top. Marvel at the view! Contemplate how the hell you will get down! The descent is possibly one of the scariest parts of this climb. Head north towards the mountains

GRADE: 0

### G02|853a87dedae7_c0

**Q46. La Sportiva vs Scarpa climbing shoes comparison**

> Topic: Scarpa Instinct VS or VSR. Love the VSRs. Softer more comfortable shoe yet downturned. They smear exceptionally well and edge a lot better than one might think. All the other shoes in my climbing bag are La Sportivas but these have earned their place. I weigh 165 for a bigger guy they might not work as well Two female climbing partners who don t weigh Much bought them and love them.

GRADE: 2

### T02|b2b75e2ee7fb_c0

**Q47. What is the best hangboard routine for intermediate climbers?**

> Topic: Bachar Ladder Help. So for John s case isn t a Bachar Ladder as poor an option as a hangboard? I m finally pushing into 5.11 territory after years of climbing myself and have just started to hangboard in the last year - I can tell that I would have quickly injured myself if I d started earlier in my climbing career. The ladder seems like a recipe for similar disaster for someone just getting into climbing - wouldn t a regime of pull-ups on jugs (say on ahem a hangboard) be a better option? Ps the best training for climbing is climbing. If you can t get outside build a woody :))

GRADE: 0

### S08|f21f16f9fe58_c0

**Q48. Fatal climbing accidents caused by equipment failure**

> Shawangunks Annual Summary Location: New York, Mohonk Preserve Type: Accident Reports Author: Andrew Bajardi In 2016 at the Mohonk Preserve there were 21 climbing-related incidents, including both injury and illness . Sixteen accidents required technical rescues. Seven of the accidents were caused by a belay system failure, while four were caused by inadequate protection. Three climbers suffered from heat-related illness. A climber sustained an unusual injury while on Star Action, a 5.10 in the Trapps. The climber fell while leading and collided with a non-locking carabiner attached to the last gear placement. The carabiner impaled the ankle and subsequently arrested the climber upside down from the heel. The climber removed the pierced carabiner and was lowered to the ground. ( Editor s note: A first-person report from this incident can be found here. ) Two climbers sustained multiple s

GRADE: 1

### G05|298928722832_c0

**Q49. Review of Black Diamond ATC Guide belay device**

> Topic: Retiring BD Guide. I would add that the teeth only provide some extra friction and are not necessary for safety. The original ATC is smooth and toothless. However if the teeth are worn other areas of the device may be worn too necessitating a replacement. Edit: It sounds like you aren't sure or don't trust your judgment so just buy a new one. Black Diamond ATC (not the "ATC Guide") ATC XP (added teeth to the regular ATC) ATC Guide

GRADE: 1

### G08|cf9eef90106c_c0

**Q50. How to choose quickdraws for sport climbing?**

> Topic: Trango Phase 18cm Quickdraws. One of my favorite all around quickdraws. Sure if your only climbing sport there are better (petzl spirit djinn etc) but for the price the phase can't be beat.

GRADE: 0

### S01|a11eaf2aabd8_c0

**Q51. What are the most common lead climbing accidents?**

> Various Falls on Rock, Mostly No or Inadequate Protection, New York, Shawangunks Type: Accident Reports VARIOUS FALLS ON ROCK, MOSTLY NO OR INADEQUATE PROTECTION New York, Shawangunks In 1989, there were 19 climbing accidents. They can be divided into three categories: (1) solo climbers or boulderers, who use no rope; (2) lead climbers; and (3) those climbers being seconded. The great majority of climbing accidents involve lead climbers. Of these, seven leaders fell, and either pulled protection or had placed none, and hit either the ground or a ledge. Six accidents involved a leader falling a moderate distance, six meters or less, with protection holding yet an injury resulting. Perhaps the most interesting category of injuries is those that happened to climbers coming second. Seconding is usually considered to be relatively safe, so injuries here are notable. All three injuries in this

GRADE: 3

### R05|44cde5b750fc_c0

**Q52. What are some classic 5.12 routes in Smith Rock Oregon?**

> Mojomatic is a 5.12a Sport climbing route. located in (zz) Upper Gorge Smith Rock Central Oregon Oregon. rated 4.0 stars. 1 pitches. Excellent arete and face moves; a classic basalt line.. Protection: bolts

GRADE: 3

### S04|4878d49b6c67_c0

**Q53. How to avoid rockfall injuries while climbing outdoors?**

> Topic: Video of massive rockfall Evolène Switzerland. In the first video I wonder how the photographer happened to be next to the site right when the rockfall occurred. Must have been scary!

GRADE: 1

### S09|867f3b0166a2_c0

**Q54. How dangerous is free soloing?**

> Topic: downclimbing a TR solo route. Yep the only reason I say unsafe is trying to be in a solid stance every time the device needs to be moved. More often than not bringing one of those down is a two handed operation making a fall much more likely. For me any soloing device or not is a NO FALL situation. The backup "pro" is there for piece of mind and just in case... In my experience if you find a good rope/device combination then it is usually hands-free but sometimes requires 1 hand to futz around with it.

GRADE: 0

### G01|538140d73bbe_c0

**Q55. What is the best belay device for multi-pitch climbing?**

> Topic: What belay device should you use?. GriGri for single pitch and ATC Guide for multi

GRADE: 3

### R08|372000766d38_c0

**Q56. Suggest a beginner-friendly climbing route in Utah**

> Easier for a Camel is a 5.5 tr climbing route. located in Utah > Southwest Utah > Saint George > Prophesy Wall

GRADE: 3

### G05|d0bce6a755ab_c0

**Q57. Review of Black Diamond ATC Guide belay device**

> Topic: Black Diamond Belay Patent. I wonder if it will work as a lead solo device

GRADE: 1

### S04|d7a8e82e5638_c1

**Q58. How to avoid rockfall injuries while climbing outdoors?**

> are much more forgiving and don t require spotting at all. In fact, gyms now teach an entirely different way to fall. Outdoors, a spotter s job is to prevent a boulderer from falling backward or hitting their head. Indoors, the emphasis is on landing on your feet and rolling backward onto the pads. My question Does this create bad habits when transitioning to outdoor bouldering that can cause harm? Does the lack of spotting in gyms reduce an important outdoor skill? Could this increase the risk of more serious injuries long-term? Curious to hear thoughts from people who climb both indoors and outdoors.

GRADE: 0

### R04|a8d16ea30dba_c0

**Q59. Recommend a multi-pitch route in Yosemite for intermediate climbers**

> Commitment is a 5.9 Trad climbing route. located in First Tier Five Open Books Yosemite Falls Area Valley North Side Yosemite Valley Yosemite National Park California. rated 3.4 stars. 3 pitches. This climb is in the next book right of "Munginella". Either climb a curving corner or a crack to the right to a belay. Climb unprotected face to the corner and continue to a tree. Continue up to a roof traverse under it to it's end and follow the corner to the top. Descend to the left. be careful not to knock rocks off onto climbers below.. Protection: Pro to 3".

GRADE: 3

### T02|88fa8997437b_c0

**Q60. What is the best hangboard routine for intermediate climbers?**

> Topic: Hangboard for Backpacking. I'm going on a thru-hike for about a month but I want to stay in as best of climbing shape as I can. I want to bring a portable hangboard with me but it cant be ridiculously heavy since I will be carrying it. Does anyone have any good suggestions for portable hangboards?

GRADE: 0

### S03|68a86531716b_c1

**Q61. What causes rappelling accidents in climbing?**

> and Robert Stock. Our regrets to anyone we ve missed. United States: Everything goes in cycles, which in the case of causes for climbing accidents is unfortunate. Nothing could be more illustrative of this than the category Rappel Failure/Error. The number of reports in this category had been in a fairly steady decline, with some spikes, the average having been five per year for the past decade. The two most common errors in the earlier years were rappelling off the end of one s rope and having the rappel anchor fail. These were corrected by tying a knot in the end of the rappel rope and having more than one anchor point if the primary protection is not deemed to be bomb proof. Last year there were twelve reports and this year there are fifteen that are attributed to rappel problems. They are mostly of a different nature than in the past, and at least half of them occurred on top-rope cl

GRADE: 3

### T03|ebb94f64d7d2_c0

**Q62. How do I improve my footwork on overhangs?**

> Topic: Resoling Evolv Shaman's. Yeah...XS Edge is probably the least sensitive rubber on the market. It's probably the thinness of the rubber that you're used to which will happen (to some extent) as the rubber wears down and you break in the shoe. Edge is good at one thing and one thing only...standing on tiny edges although it probably is the best for that. That said if you like sensitivity (I do) go for a softer shoe. XS Grip is much more sensitive though my favorite is still Stealth C4. I don't see why climbing for 10 months automatically means you're not ready for a soft shoe...I would argue the opposite (after arguing that timelines are stupid). Climbing in a softer shoe you will get stronger as the shoe will do less of the heavy lifting for you and your footwork will improve as you will feel the rock better and can develop better proprioception.

GRADE: 0

### T06|d1e4be537375_c0

**Q63. What is a good weekly training schedule for V5 bouldering?**

> My First Boulder is a V0 bouldering climbing route. located in Georgia > Hitchiti Boulders > Boulder 6

GRADE: 0

### R02|c97f2f3a6cde_c0

**Q64. What are some good beginner trad routes in Red River Gorge?**

> Party Time is a 5.7 Trad climbing route. located in Fortress Wall Northern Gorge Red River Gorge Kentucky. rated 3.4 stars. 2 pitches. 110.0 feet long. One of the nicest moderate trad routes in the Red and it takes good gear if you have it. The first pitch takes a few larger cams (perhaps to 4") and the second pitch a standard light rack. Bolted belay/rap anchors make this climb a good first multi-pitch. The ledge where the anchors are to end the first pitch is pretty small and can hold 4 people max.. Protection: A few large cams for the leaning start then head for the top on a standard rack. With a few longer slings this could be done in a single pitch. But the bolted anchors make breaking it up quite easy.

GRADE: 3

### R04|841d1cd0e100_c0

**Q65. Recommend a multi-pitch route in Yosemite for intermediate climbers**

> Crashline is a 5.11b Trad climbing route. located in u. Elephant Rock Lower Merced River Canyon Yosemite Valley Yosemite National Park California. rated 3.2 stars. 1 pitches. 120.0 feet long. A straightforward crack with a low crux. Clean and elegant.. Protection: Extra small gear. Easy to eyeball it from below. Can break into two pitches at an intermediate tree if desired.

GRADE: 3

### G05|20261f49f867_c0

**Q66. Review of Black Diamond ATC Guide belay device**

> Topic: DMM Pivot Belay Device. Fellow ATC guide user I don't think it's worth the money for a slight variation of something we already have. It might be a bit better in guide mode - but if guide mode was important to me I'd have gotten a King gigi already.

GRADE: 0

### S05|322df6fd83e1_c0

**Q67. What happened in climbing accidents at Red River Gorge?**

> Fall on Rock, Protection Pulled Out, Inadequate Protection, Kentucky, Red River Gorge, Muir Valley Type: Accident Reports FALL ON ROCK, PROTECTION PULLED OUT, INADEQUATE PROTECTION Kentucky, Red River Gorge, Muir Valley On April 22, a male climber (40) was nearing the top of a short dihedral on a trad route called Short and Sweet (5.7), located at the Practice Wall. He took a lead fall, causing his upper protection pieces to pop out, causing him to hit the ground. He sustained a crushed vertebra. He was carefully packaged onto a spine board, placed in a litter, and delivered to an awaiting helicopter. Analysis This accident was caused by tenuously placed gear. The climber commented that he should have put in more pieces. (Source: Rick Weber)

GRADE: 3

### G04|4556359c4016_c0

**Q68. Best harness for beginners?**

> Topic: Edelrid Orion Harness. I looked around for a new harness a couple months ago and tried on a lot of different brands and models. For me nothing came close to the Orion... really comfortable adjustable and well made. I've used it trad climbing and for some sport. I've been climbing a long time and I think its the best harness I've owned. Highly recommended. I can't speak to its longevity but I've never had a more comfortable harness.

GRADE: 3

### R06|1ae610c7e175_c0

**Q69. Find me a 5.8 sport route in Joshua Tree**

> Bubba-Do is a 5.8+ sport climbing route. located in California > Joshua Tree National Park > Central Joshua Tree > Echo Rock Area > Cuckoo's Nest, The. First ascent: Alan Bartlett. F&K

GRADE: 3

### G03|dce12256d707_c0

**Q70. What climbing rope should I buy for sport climbing?**

> Topic: Preferred Rope(s). If you just want names of Brands I think they all have been said here already maybe add Beal to the list. If you really want good advice tell us a little more about what kind of climbing you will use it for. Sportclimbing multi-pitch trad Alpine? top-roping?? Its very easy to over buy when it comes to rope. Remember you will retire it someday and you don't want to feel like you wasted money buying something you never used the extra features that you payed extra for.. Start comparing ropes from the bottom of the list not the top (based on feature/price) buy only what you need for now. You will probably buy another later so you can upgrade then if you move on.. My 2 cents..

GRADE: 2

### T07|8827a855a0c5_c1

**Q71. How do I build endurance for long multi-pitch routes?**

> keep climbing upward on indistinct terrain for maybe 15 meters more. There is drag on this pitch if you don't take long slings and manage gear placements. Rossiter suggests double ropes in his book.. Protection: A standard light rack - 1 set stoppers one set cams a few Tricams if desired and a few draws for the "so-so" pins that protect the crux moves. Bring some LONG slings to avoid drag on this zig-zag line.

GRADE: 0

### S04|621546c41c9d_c0

**Q72. How to avoid rockfall injuries while climbing outdoors?**

> Rockfall Location: Oregon, Mt. Washington Type: Accident Reports Author: Corvallis Mountain Rescue Unit In the evening of October 11, a 20-year-old solo climber was hit in the head by rockfall while rappelling the summit block of this 7,795-foot volcano . This caused her to fall about 15 feet and badly injure a knee. The climber was able to descend to about 6,200 feet, but could not continue. She called for help at around 11:25 p.m., and her cell phone battery then died. Rescuers were able to locate her around 7 a.m. Because of the nature of her injuries and the rugged location, she was airlifted from the scene by Oregon Army National Guard helicopter. ( Source: Corvallis Mountain Rescue Unit and news reports .) ANALYSIS A similar accident occurred on Mt. Washington in July, when a climber was hit by rockfall while rappelling the upper mountain. Both incidents suggest the importance of u

GRADE: 1

### G06|39656573211c_c0

**Q73. What helmet should I use for outdoor climbing?**

> Topic: Crash Helmet/ Climbing Helmet. I don't understand if by asking for a "crash" helmet you are looking for something different than your standard climbing helmet? Check out outdoorgearlab.com for detailed reviews. My 0.02: -make sure it fits well for all your intended uses (i.e. no hat hat etc) -good protection for your uses or at least you recognize the caveat emptor of helmet certifications etc. I like foam helmets i.e. Petzl Meteor because they are light and offer good occiput protection (in theory) during an upsidedown fall. They are less durable than plastic shell helmets. For ice multipitch and alpine I now always wear a Meteor III (or foam helmet of your choice) because of the comfort fit and protection. Unless I'm cragging I often put my Meteor III helmet on at the car and leave it on all day. I use it skiing as well on trips where I don't want to pack a separate ski helmet. 

GRADE: 3

### T10|82e8ac39d893_c0

**Q74. Should I campus board as a beginner?**

> Topic: Hangboard Designs?. That would be fine as well- 15mm might be too small for a beginner. Another thing you could do would be to get both slots routed to 25mm but get a couple 5x30mm strips that you could stick in the back of the slots to adjust the depths.

GRADE: 1

### G10|70f5118fa8d8_c0

**Q75. Are assisted braking belay devices safer than tube style?**

> Topic: Edelrid Mega Jul?. Stephen I think reality is a bit more nuanced than you are suggesting. Before saying more I should add that although my age qualifies me as one of the fuddy-duddy's whose opinions you are trying to disqualify I have been a proponent and almost continual user of some of the assisted locking devices for many years now (except in the gym where I still use a modern tube.) I do think it is a good idea to keep both eyes open about both the advantages and the drawbacks of various techniques however and when you relegate tubes to nothing more than "a rite of passage " I think you've oversimplified for at least four reasons. (1) At least some of the assisted braking devices have performance that may be severely affected by environmental conditions. Tubes are more reliable and may be the only choice for some circumstances. Climbers who might find themselves in adverse sit

GRADE: 2

### T09|6c3a8346c955_c0

**Q76. How to overcome a climbing plateau at 5.11?**

> Topic: Best books on performace to climb 5.12?. How to climb 5.12: Climb a lot. At least 4x a week. Climb with different groups Climb with 5.12 climbers Don't get fat

GRADE: 3

### S04|8a01c8865e72_c1

**Q77. How to avoid rockfall injuries while climbing outdoors?**

> nce the last rockfall-related fatality in the park, when climber Peter Terbush was killed by a rockfall from Glacier Point on June 13, 1999. ANALYSIS Most rockfalls in Yosemite occur in the winter and early spring, during periods of intense rainfall, snowmelt, and/or subfreezing temperatures, but large rockfalls like these ones from El Capitan have occurred during periods of warm, stable weather. How can climbers address rockfall risk? Unfortunately there are no hard or fast rules, but rockfall areas are often active for many hours, days, or even months, so avoid climbing in recent rockfall zones. The progressive nature of the El Cap rockfalls, with several events from the same location, has also been seen at other locations in Yosemite, including Middle Brother and the Rhombus Wall. Fresh talus and/or damaged vegetation at the base of your intended climb are good indicators of recent ac

GRADE: 1

### R10|dc738084a909_c0

**Q78. Recommend a V0-V2 bouldering area near San Francisco**

> Klinghoffers Roof is a V3 bouldering climbing route. located in California > San Francisco Bay Area > Castle Rock Area > * Castle Rock Area Bouldering > Klinghoffers

GRADE: 1

### R10|8fb7618f0f73_c0

**Q79. Recommend a V0-V2 bouldering area near San Francisco**

> How about that is a V2 bouldering climbing route. located in California > San Francisco Bay Area > Wine Country > Nut Tree Boulders, The > Woodcrest Boulders > Woodcrest Ridge Boulders > Tidbit Boulder. First ascent: Aaron Rough?

GRADE: 3
