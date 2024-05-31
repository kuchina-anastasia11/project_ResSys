## HAHA.ru

Telegram bot with a job recommendation system.

Link: [HAHA.ru](https://t.me/haha_project_bot) (may be temporarily unavailable)




![Bot screenshot](./data/photo_2024-03-29_20-17-58.jpg)


## Short description
A Telegram bot that offers personalized job recommendations on the Russian market for each user.


## How interaction with the bot works

1. The user sends the resume to the bot as a PDF file or manually inputs the data.
2. Based on this data, an initial selection of job vacancies is created.
3. The user can interact with the suggested job vacancies. Available actions include:
    * Apply
    * Add to favorites
    * Next vacancy
4. Subsequent recommendations take into account the user's interaction with previously suggested job vacancies.


## Data
The datasets used can be found in [data/data.txt](./data/data.txt).


## Try our model
If you want to try our model for your predictions, import it like this:

```shell
pip install -r requirements.txt

# Initial recommendation model (based on resume data)
from pipelines.initial_recommendation import initial_recommendation

r = initial_recommendation(vacancy_data)
vacancy_ids = r.recommend(vacancy_name, salary, region_id, work_experience, key_skills)


# Secondary recommendation model (taking previous job interactions into account)
from pipelines.predict import predictor

r = predictor(vacancy_data)
vacancy_ids = r.recommend(vacancy_name, region_id, work_experience, action_type, vacancy_id_list, key_skills)
```


## Team

- [Anastasia Kuchina](https://github.com/kuchina-anastasia11)
- [Svetlana Lundina](https://github.com/Vambassa)
- [Anna Maksimova](https://github.com/anpalmak2003)
