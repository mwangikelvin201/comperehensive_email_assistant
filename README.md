# comperehensive_email_assistant
![My image](https://github.com/mwangikelvin201/comperehensive_email_assistant/blob/c47d06349ca6193449b75595a86682717fe0001d/AI-Email-Assistant-2.png)
## This is an ongoing comprehensive mail assitant project that uses OpenAI LangChain, LangGraph,OpenAI and Pinecone for custom data retrieval

### What it does...
## assistant
Draft custom company's documents like memos and receive and reply to emails with personalized company's Data

## Lessons learnt along the process

1. Prompting is notably the most important aspect/skill in any Large Language model/AI project. It is the core instructing and governing factor to the app.
2. Te prompting should always be ReActive. Start with little prompts and build on them noticing the imrovements as well as the mistakes. This helps in rectifying any poor outputs from the AI agent


## Email_monitor


This is a new advancement...I decided to keep it in a whole separate new notebook(email_monitor.py).

This one goes through the emails and  evaluates if an email is eligible for an automatic response using an evaluator function defined in the code . 

The email has to be company related eg " what are the company's policies on customer data privacy", so that it can be replied using the information contained inside the vector database.

When running,it refreshes after 60 seconds(1 minute) and monitors for any new email. It evalautes the nature of the email and if the email is company related, It queries the vector database and replies the email accordingly.



### Challenging task....

Reduce the api calls by the app to the  llm in order to reduce the overall cost...but maintain the accuracy and the general functionality of the app


