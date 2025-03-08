import streamlit as st

@st.fragment(run_every=15)
def display_task_status():
    #print('display task status')
    if (('task_queue' not in st.session_state) or 
        (len(st.session_state.task_queue) == 0)):
        st.info('No tasks running')
        return
    
    for idx, task in enumerate(st.session_state.task_queue):
        status = task.status()
        if task.active():
            state = 'running'
            msg = 'checking again in 15 seconds...'
        elif status['state'] == 'COMPLETED':
            state = 'complete'
            msg = 'Completed Successfully'
        elif status['state'] == 'FAILED':
            state = 'error'
            msg = 'Failed.  Contact support.'
            st.toast('Save failed.  Contact support.', icon=":material/error:")
        elif status['state'] == 'CANCELLED':
            state = 'complete'
            msg = 'Cancelled.'
        else:
            state = 'error'
            msg = 'ended with status '+status['state']

        st.status(f'Task {idx}: Saving labels...'+msg, state=state)