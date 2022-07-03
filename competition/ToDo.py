# To Inform/Ask.
A. There is now a PR1487 for the competition code. Consists online training, submission, and evaluation for track-1. If anyone wants to add changes to my working branch, my current working branch for the competition is `comp-2`.

B. Scoring
@Soheil 
1. Please check the scoring function implemented in:
i) SMARTS/competition/evaluation/evaluate.py
ii) SMARTS/competition/evaluation/metric.py
iii) SMARTS/competition/evaluation/score.py
2. The scales of each score component, i.e., `completion`,`time`,`humanness`,`rule`, are different. Thus it is difficult to combine them into a single overall score.
3. Participants should strive to minimise each score component, i.e., minimize the cost function. The lower the better.
4. Consider computing the overall rank by ordering according to priority of each score component.

C. Offline Learning
For those who working on offline learning portion and track 2 of the competition.
1. Please write track-2 instructions for participants. Provide explanation on (i) interfaces that need to be adhered to, (ii) scenarios, (iii) datasets, and (iv) any necessary code examples, for the participants to get started with track-2.
2. Please verify the train, submit, evaluate, process explained for track-2 in point 1 above.
3. Make the necessary pull request for track-2.

# To Do
# 1. Traci Error.
# 2. Codalab uploading.
# 3. Cut-in scenario does not work.
# 4. Overtake scenario does not have enough diversity.
# 5. Retrain model.
# 6. Different traffic rou.xml file is built each time. For evaluation, fix the built rou.xml files to ensure same score for same models.
# 7. Preferably, redirect participants to get the competition starter code from SMARTS Github repository.
# 8. Use requirements.txt to install dependencies from paticipant submission for evaluation.
# 9. Rank scores in order of priority i.e., completion > time > humanness > rules.
