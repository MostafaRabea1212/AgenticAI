from src.agents.agents import build_search_agent , build_reader_agent , writer_chain,critic_chain

def run_reaserch_agent(topic : str) -> dict :

    state={}

    #search agent working 
    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    search_agent=build_search_agent()
    search_result= search_agent.invoke({
        "messages" : [("user", f"Find recent, reliable and detailed information about: {topic} , Return their titles, URLs, and short descriptions.")]
    })

    state["search_results"] = search_result["messages"][-1].content


    #step 2 - reader agent 
    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    reader_agent=build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [
            (
                "user",
                f"""
                Research topic: {topic}

                Extract up to 3 relevant URLs.
                Use scrape_url to scrape them.
                Summarize the important findings.

                Search results:
                {state['search_results'][:1000]}
                """
            )
        ]
    })
    state["scraped_content"] = reader_result["messages"][-1].content
    print("\nscraped content: \n", state['scraped_content'])

    #step 3 - writer chain 
    print("\n"+" ="*50)
    print("step 3 - Writer is drafting the report ...")
    print("="*50)

    research_combined = (
        f"search_result :\n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic" : topic , 
        "research" :research_combined
    })
    print("\n Final Report\n",state['report'])

    #critic report 
    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
        "report" : state['report']
    })

    print("\n critic report \n", state['feedback'])

    return state
