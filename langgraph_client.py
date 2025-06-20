from raptor.graph import workflow as raptor_workflow
from crag.graph import workflow as crag_workflow

def run_raptor(url, max_depth, question, thread_id):
    input_data = {"url": url, "max_depth": max_depth, "question": question}
    result = raptor_workflow.invoke(input_data, config={"configurable": {"thread_id": thread_id}})
    return result["answer"]

def run_crag(urls, question, thread_id):
    input_data = {"urls": urls, "question": question}
    result = crag_workflow.invoke(input_data, config={"configurable": {"thread_id": thread_id}})
    if result:
        return result.get("answer", "")
    return ""