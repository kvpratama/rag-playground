from raptor.graph import workflow

def run_workflow(url, max_depth, question, thread_id):
    input_data = {"url": url, "max_depth": max_depth, "question": question}
    result = workflow.invoke(input_data, config={"configurable": {"thread_id": thread_id}})
    return result["answer"]