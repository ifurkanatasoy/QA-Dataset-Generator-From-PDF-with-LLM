# QA Dataset Generator from PDF with LLM

>[!NOTE]
>*Since the PDF documents are in Turkish, the prompts used are also in Turkish.*
>
## Summary of the Process
Using Gemini to create a Turkish dataset, 

1. Divide the content from the PDF into chunks of length 1028 characters, with 128 characters overlapping.
2. Ask Gemini to generate questions for each chunk. 
2. Send the questions and relevant chunks one by one back to Gemini, and ask it to answer each question based on the corresponding chunk.

## Fixing the Response Format
Since LLMs do not always produce responses in the same format, we could not receive the questions and answers properly within a Python code. We have solved this problem with a different approach. For example, a default response format looks like this:

>Of course, I can produce questions and answers as you wish:
> 
>Question1?
> 
>Answer1
> 
>Here is another question and answer:
>
>
>Question2?
> 
>Answer2
> 
>I hope it was useful.

Another response format can be:

>Of course I can make it in the format you want.
> 
>Question1? Answer1
> 
>Question2? Answer2

As a solution to this problem, we changed the prompt so it displays question-answer pairs between certain tags. Using our format here's an example respose from Gemini:

>Of course, I can produce questions and answers as you wish:
> 
>[q] Question1 [/q]
> 
>[a] Answer1 [/a]
>
>[q] Question2 [/q]
> 
>[a] Answer2 [/a]
>
>...

>[!NOTE]
>[q] stands for "Question"
> 
>[a] stands for "Answer"

Thanks to our solution, now we can extract all the questions and answers using Regex (Regular Expression) without missing on any data.
