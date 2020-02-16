## happy path
* greet
  - utter_greet
* mood_great
  - utter_happy

## sad path 1
* greet
  - utter_greet
* mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* affirm
  - utter_happy

## sad path 2
* greet
  - utter_greet
* mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* deny
  - utter_goodbye

## say goodbye
* goodbye
  - utter_goodbye

## bot challenge
* bot_challenge
  - utter_iamabot

## colour challenge
* colour_challenge
  - utter_favouritecolour

## weather challenge
* weather_challenge
  - utter_weather

## mood challenge
* mood_challenge
  - utter_mood

# mood challenge path 2
* mood_challenge
  - utter_mood
* mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* affirm
  - utter_happy
  
# mood challenge path 3
* mood_challenge
  - utter_mood
* affirm
  - utter_happy

## action challenge
* action_challenge
  - utter_action

## action challenge path 2
* action_challenge
  - utter_action
* affirm
  - utter_happy 

## action challenge path 3
* action_challenge
  - utter_action
* deny
  - utter_rebuttal 

## rebuttal
* rebuttal
  - utter_rebuttal

## apoligise
* apoligise
  - utter_apology_accepted

## name challenge
* name_challenge
  - utter_name
* name_response
  - action_respond_name
  - utter_name_response
