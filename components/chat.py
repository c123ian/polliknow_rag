import logging
from fasthtml.common import *
from components.assets import send_icon

def chat_input(disabled=False):
    return Input(
        type="text",
        name="msg",
        id="msg-input",
        required=True,
        placeholder="Type a message",
        hx_swap_oob="true",
        autofocus="true",
        disabled=disabled,
        cls="!mb-0 bg-zinc-900 border border-zinc-700 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-zinc-500 disabled:bg-zinc-800 disabled:border-zinc-700 disabled:cursor-not-allowed rounded-md",
    )

def chat_button(disabled=False):
    return Button(
        send_icon(),
        id="send-button",
        disabled=disabled,
        cls="bg-green-500 hover:bg-green-600 text-white rounded-md p-2.5 flex items-center justify-center border border-zinc-700 focus-visible:outline-none focus-visible:ring-zinc-500 disabled:bg-green-800 disabled:border-green-700 disabled:cursor-not-allowed",
    )

# Update the chat_form component to include the session ID
def chat_form(disabled=False, session_id=None):
    return Form(
        Input(
            type="text",
            name="msg",
            id="chat-input",  # Important: ID used in JS
            placeholder="Type your message...",
            disabled=disabled,
            cls="input input-bordered w-full",
            autocomplete="off",
        ),
        Input(
            type="hidden",
            name="session_id",
            value=session_id,  
        ),
        Script(
            """
            document.body.addEventListener('htmx:wsAfterSend', function() {
                document.getElementById('chat-input').value = '';
            });
            """
        ),
        cls="flex gap-2",
        hx_ws="send",  
    )


def chat_message(msg_idx, messages):
    msg = messages[msg_idx]
    # Improved content class to better handle long messages
    content_cls = f"px-2.5 py-1.5 rounded-lg max-w-lg {'rounded-br-none border-green-700 border' if msg['role'] == 'user' else 'rounded-bl-none border-zinc-400 border'}"
    
    # Log the message content for debugging
    logging.info(f"Rendering message {msg_idx} from {msg['role']}: {msg['content'][:100]}...")
    
    return Div(
        Div(msg["role"], cls="text-xs text-zinc-500 mb-1"),
        Div(
            msg["content"],
            # Updated styling to handle long text better
            cls=f"bg-{'green-600 text-white' if msg['role'] == 'user' else 'zinc-200 text-black'} {content_cls} whitespace-pre-wrap break-words",
            id=f"msg-content-{msg_idx}",
        ),
        id=f"msg-{msg_idx}",
        cls=f"self-{'end' if msg['role'] == 'user' else 'start'} max-w-[85%]",
    )

def chat_window(messages):
    return Div(
        id="messages",
        *[chat_message(i, messages) for i in range(len(messages))],
        cls="flex flex-col gap-2 p-4 h-[55vh] overflow-y-auto w-full",
    )

def chat(session_id, messages):
    # Log all messages for debugging
    logging.info(f"Rendering chat component with session_id: {session_id}")
    for idx, msg in enumerate(messages):
        logging.info(f"Message {idx} from {msg['role']}: {msg['content'][:200]}...")
        if len(msg['content']) > 200:
            logging.info(f"... message continues (total length: {len(msg['content'])} chars)")
    
    return Div(
        chat_window(messages),
        chat_form(disabled=False, session_id=session_id),  # Pass session_id to form
        Script(
            """
            function scrollToBottom(smooth) {
                var messages = document.getElementById('messages');
                messages.scrollTo({
                    top: messages.scrollHeight,
                    behavior: smooth ? 'smooth' : 'auto'
                });
            }
            window.onload = function() {
                scrollToBottom(true);
            };

            const observer = new MutationObserver(function() {
                scrollToBottom(false);
            });
            observer.observe(document.getElementById('messages'), { childList: true, subtree: true });
            
            // Add a resize observer to handle dynamic content
            const resizeObserver = new ResizeObserver(() => {
                scrollToBottom(true);
            });
            
            // Observe all message contents
            document.querySelectorAll('[id^="msg-content-"]').forEach(el => {
                resizeObserver.observe(el);
            });
            """
        ),
        hx_ext="ws",
        ws_connect="/ws",
        cls="flex flex-col w-full max-w-3xl border border-zinc-700 h-full rounded-md outline-1 outline outline-zinc-700 outline-offset-2 relative",
    )
