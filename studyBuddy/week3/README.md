### To run the code follow the same setup as week 2 README.md  

Check out the [Week 2 README here](../week2/README.md).

- In addition, set your `TAVILY_API_KEY` in a `.env` file in the root folder.

> TAVILY_API_KEY= "******************************"

# Testing using various test cases

### 1. Tool Use: Tavily Search (Web Search API)
Q. What are the latest developments in quantum computing as of June 2025?
![Tavily Search](image.png)

### 2. Tool Use: Wikipedia (Factual Summary)

Q. Who was Max Planck and what is he famous for?
![wikipedia](image-1.png)

You can see that when the first time I didnt added who was for the tool selector keyword(thats why showing Selected tool: none), the response was not rich, and the 2nd response is more rich because of using wikipedia

### 3. Tool Use: Calculator

Q. Calculate 4+568
![alt text](image-2.png)

### 4. RAG Only (No Tool Needed)

Q. What is the role of quantum tunneling in semiconductor devices?
![alt text](image-3.png)

### 5. Out-of-Scope Query 

Q. What is the capital of Mars?
![alt text](image-5.png)

### 6. Memory/History Reference (Multi-turn)

Q1. What is wave-particle duality?

Q2. Can you explain how it relates to the photoelectric effect?
![alt text](image-4.png)