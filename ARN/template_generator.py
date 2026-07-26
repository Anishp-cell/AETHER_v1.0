"""
Template-based dataset generator for ARN.
Generates diverse training pairs using pure Python randomization.
Every argument value is an EXACT substring of the user_prompt.
Includes support for technical vocabulary, slang, corrections, and new compounds.
"""

import json
import os
import random

# ── Contact names & relations ───────────────────────────────────────
CONTACTS = [
    "Mom", "Dad", "Bhai", "Didi", "Amma", "Papa",
    "John", "Sarah", "Priya", "Rahul", "Ankit", "Sophie",
    "Mr. Davis", "Mrs. Sharma", "Dr. Patel", "Prof. Gupta",
    "Ravi", "Neha", "Arjun", "Aarav", "Meera", "Kavya",
    "Alex", "Emma", "Liam", "Zara", "Rohan", "Ishaan",
    "Tanya", "Vikram", "Sneha", "Kiran", "Amit", "Pooja",
    "Sahil", "Nisha", "Dev", "Aanya", "Siddharth", "Kriti",
    "director", "manager", "friend", "bro", "sister", "boss"
]

RELATIONS = [
    "my mom", "my dad", "my brother", "my sister",
    "my friend", "my boss", "my manager", "my colleague",
    "my roommate", "my cousin", "my uncle", "my aunt",
    "my wife", "my husband", "my girlfriend", "my boyfriend",
    "my teacher", "my mentor", "my neighbour", "my director",
    "my bro", "my sis"
]

# ── Devices & locations ─────────────────────────────────────────────
DEVICES = ["lights", "fan", "AC", "heater", "lamp", "bulb", "TV"]
ROOMS = ["bedroom", "kitchen", "living room", "bathroom", "hall", "balcony", "study room"]
DEVICE_ACTIONS = ["turn on", "turn off", "switch on", "switch off"]

# ── Apps ────────────────────────────────────────────────────────────
APPS = ["Notepad", "Word", "VS Code", "Chrome", "Excel", "PowerPoint",
        "Sublime Text", "Terminal", "Notes", "TextEdit", "Obsidian"]

# ── URLs / websites ─────────────────────────────────────────────────
WEBSITES = [
    ("YouTube", "youtube.com"), ("LinkedIn", "linkedin.com"),
    ("Instagram", "instagram.com"), ("Twitter", "twitter.com"),
    ("Netflix", "netflix.com"), ("Gmail", "gmail.com"),
    ("GitHub", "github.com"), ("Reddit", "reddit.com"),
    ("Amazon", "amazon.in"), ("Flipkart", "flipkart.com"),
    ("Google Drive", "drive.google.com"), ("Spotify", "spotify.com"),
    ("WhatsApp Web", "web.whatsapp.com"), ("ChatGPT", "chat.openai.com"),
    ("Stack Overflow", "stackoverflow.com"),
]

# ── Search queries ──────────────────────────────────────────────────
SEARCH_QUERIES = [
    "best restaurants near me", "how to learn Python fast",
    "latest iPhone price in India", "weather forecast for tomorrow",
    "paneer butter masala recipe", "top 10 movies 2025",
    "how to fix a leaking tap", "machine learning roadmap",
    "best budget laptops under 50000", "train tickets from Delhi to Mumbai",
    "what is quantum computing", "symptoms of vitamin D deficiency",
    "best places to visit in Manali", "how to make cold coffee at home",
    "IPL match schedule 2026", "healthy breakfast ideas",
    "how to center a div in CSS", "best free coding bootcamps",
    "difference between AI and ML", "petrol price today",
    "nearest pharmacy open now", "how to invest in mutual funds",
    "best gym exercises for beginners", "how to remove dark circles",
    "who built the Great Pyramid of Giza", "the latest stock price of NVIDIA",
    "setup errors"
]

# ── WhatsApp messages ───────────────────────────────────────────────
MESSAGES = [
    "I will be late today", "coming in 10 minutes",
    "can you send me the notes", "happy birthday bro",
    "let's meet at 5 pm", "don't forget to bring the charger",
    "I reached home safely", "call me when you are free",
    "meeting is postponed to tomorrow", "please pick up milk on your way home",
    "the project deadline is extended", "are you coming to the party tonight",
    "I finished the assignment", "thanks for your help today",
    "can we reschedule our call", "good morning have a great day",
    "I just left the office", "send me your location",
    "the food was amazing", "I will handle it don't worry",
    "please check your email", "where are you right now",
    "I need some help with this", "congratulations on your promotion",
    "hi", "I'm late", "the deployment went live smoothly"
]

# ── Reminder messages ───────────────────────────────────────────────
REMINDERS = [
    "take my medicine", "drink water", "call the dentist",
    "submit the report", "pick up groceries", "charge my laptop",
    "water the plants", "feed the cat", "check the laundry",
    "reply to that email", "stretch and take a break",
    "prepare for the meeting", "backup my files",
    "pay the electricity bill", "order dinner",
]

# ── Deepseek reasoning queries ──────────────────────────────────────
REASONING_QUERIES = [
    "what is the integral of x squared times e to the x",
    "solve the traveling salesman problem for 5 cities",
    "explain the proof of Fermat's last theorem",
    "derive the Navier-Stokes equation from first principles",
    "what is the time complexity of merge sort and why",
    "prove that the square root of 2 is irrational",
    "explain P vs NP problem in simple terms",
    "solve this differential equation: dy/dx = 3x^2 + 2x",
    "what is the Schrodinger equation and how does it work",
    "calculate the eigenvalues of a 3x3 matrix",
    "explain how neural networks learn using backpropagation",
    "what happens inside a black hole according to general relativity",
]

# ── Script instructions ─────────────────────────────────────────────
SCRIPT_INSTRUCTIONS = [
    "a Python script that downloads all images from a webpage",
    "a script to rename all files in a folder with timestamps",
    "a Python script to convert CSV to JSON",
    "a script that monitors CPU usage every 5 seconds",
    "a Python script to merge multiple PDFs into one",
    "a script that sends an email with an attachment",
    "a Python script to scrape weather data from a website",
    "a script that generates a random password with 16 characters",
    "a Python script to resize all images in a directory to 800x600",
    "a script that backs up a folder to a zip file",
    "clone the git repository at github dot com slash production repo",
]

# ── Computer commands / targets ─────────────────────────────────────
COMPUTER_TARGETS = [
    ("open_app", "Calculator"), ("open_app", "Task Manager"),
    ("open_app", "File Explorer"), ("open_app", "Settings"),
    ("open_app", "Control Panel"), ("open_app", "Paint"),
    ("open_app", "Command Prompt"), ("open_app", "Snipping Tool"),
    ("press_key", "alt+tab"), ("press_key", "ctrl+shift+esc"),
    ("press_key", "win+d"), ("press_key", "ctrl+z"),
    ("press_key", "ctrl+s"), ("press_key", "print screen"),
    ("shell", "ipconfig"), ("shell", "dir"), ("shell", "tasklist"),
    ("shell", "systeminfo"), ("shell", "ping google.com"),
    ("shell", "rm -rf var log structural entries"),
    ("shell", "check on our local k8s cluster health status"),
]

# ── Teach new skill ─────────────────────────────────────────────────
SKILL_TEMPLATES = [
    ("auto_backup", "automatically backup my documents folder", "write a script that copies all files from Documents to a backup folder on D drive every day"),
    ("screenshot_tool", "take a screenshot and save it", "capture the current screen using Python and save it as a PNG file with timestamp"),
    ("email_checker", "check my unread emails", "use IMAP to connect to Gmail and count unread emails then show a notification"),
    ("wifi_scanner", "scan nearby wifi networks", "run netsh wlan show networks command and parse the output to list all available SSIDs"),
    ("disk_cleaner", "clean temporary files from my PC", "find and delete all files in the temp folder that are older than 7 days"),
    ("file_organizer", "organize files by extension", "sort all files in Downloads folder into subfolders based on their file extensions"),
    ("daily_greeting", "greet me every morning with weather", "fetch the weather API and speak a good morning message with the temperature"),
    ("youtube_downloader", "download YouTube videos", "use yt-dlp to download a video from a given YouTube URL to the Videos folder"),
    ("clipboard_history", "save clipboard history", "monitor the clipboard every 2 seconds and save new entries to a text file"),
    ("pomodoro_timer", "run a pomodoro study timer", "create a 25 minute countdown timer that plays a sound and shows a notification when done"),
    ("pull metrics", "fetch the local system CPU utilization every 5 seconds using psutil and append it to an analytics file", "fetch the local system CPU utilization every 5 seconds using psutil and append it to an analytics file"),
    ("deploy cluster", "run the bash script setup-env sh with full root permissions if needed", "run the bash script setup-env sh with full root permissions if needed"),
]

# ── Media control values ────────────────────────────────────────────
MEDIA_ACTIONS = [
    ("play", None), ("pause", None), ("next", None), ("previous", None),
    ("mute", None), ("unmute", None),
    ("set_volume", 20), ("set_volume", 30), ("set_volume", 50),
    ("set_volume", 70), ("set_volume", 80), ("set_volume", 100),
]

# ── Screen analysis queries ─────────────────────────────────────────
SCREEN_QUERIES = [
    "what is on my screen right now",
    "read the text on my screen",
    "what am I looking at",
    "describe what you see on my display",
    "can you read this error message on screen",
    "what does my screen say",
    "read this code on my screen and explain it",
    "what app am I using right now",
]

# ── Timer durations ─────────────────────────────────────────────────
TIMER_MINUTES = [1, 2, 3, 5, 10, 15, 20, 25, 30, 45, 60]

# ── Casual prefixes / suffixes ──────────────────────────────────────
PREFIXES = [
    "hey ", "yo ", "bro ", "dude ", "ok ", "umm ", "please ",
    "can you ", "could you ", "hey aether ", "aether ", "",
    "quickly ", "just ", "real quick ", "do me a favor and ",
    "i need you to ", "go ahead and ", "alright ", "listen ",
    "so basically ", "okay so ", "right so ", "plz ", "plzzz ",
]
SUFFIXES = [
    "", " thanks", " please", " asap", " right now",
    " bro", " dude", " for me", " will you",
    " ok?", " thx", " ty", " na", " yaar",
    " quickly", " rn",
]

DIAGNOSTIC_QUERIES = [
    "check the phone battery life",
    "check the laptop battery level",
    "chk my pc batt level 2 b 4gud",
    "check my system status",
    "what's my battery level",
    "show me CPU and RAM usage",
    "give me a system health report",
    "how much battery do I have left",
    "system diagnostics please",
    "mera system ka status batao",
    "is my laptop overheating"
]


def make_contact():
    """Return a contact name string as it would appear in the prompt."""
    if random.random() < 0.4:
        return random.choice(RELATIONS)
    elif random.random() < 0.6:
        rel = random.choice(RELATIONS)
        name = random.choice(CONTACTS)
        return f"{rel} {name}"
    else:
        return random.choice(CONTACTS)


def make_pair_get_current_time():
    templates = [
        "what time is it{sfx}",
        "{pfx}tell me the current time{sfx}",
        "{pfx}what's the time right now{sfx}",
        "bata na time kya hua hai{sfx}",
        "{pfx}check the time for me{sfx}",
        "time please{sfx}",
        "kya time ho raha hai{sfx}",
        "{pfx}I need to know the exact time{sfx}",
    ]
    t = random.choice(templates).format(pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES))
    tools = [{"name": "get_current_time"}]
    return t, tools


def make_pair_handle_smart_home():
    device = random.choice(DEVICES)
    room = random.choice(ROOMS)
    action = random.choice(DEVICE_ACTIONS)
    templates = [
        "{pfx}{action} the {device} in the {room}{sfx}",
        "{pfx}{action} {room} {device}{sfx}",
        "{pfx}{device} {action} in {room}{sfx}",
        "{room} ka {device} {action} karo{sfx}",
    ]
    t = random.choice(templates).format(
        pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES),
        device=device, room=room, action=action
    )
    tools = [{"name": "handle_smart_home", "arguments": {"device": room, "action": action}}]
    return t, tools


def make_pair_route_to_deepseek():
    query = random.choice(REASONING_QUERIES)
    templates = [
        "{pfx}{query}{sfx}",
        "{pfx}solve this for me: {query}{sfx}",
        "{pfx}I have a hard question: {query}{sfx}",
        "think about this: {query}{sfx}",
    ]
    t = random.choice(templates).format(
        pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES), query=query
    )
    tools = [{"name": "route_to_deepseek", "arguments": {"query": query}}]
    return t, tools


def make_pair_open_app_and_type():
    app = random.choice(APPS)
    texts = [
        "hello world", "meeting notes for today", "TODO: finish the project",
        "shopping list: eggs milk bread butter", "dear sir I am writing to inform you",
        "import numpy as np", "the quick brown fox jumps over the lazy dog",
        "reminder: call the bank tomorrow", "chapter 1 introduction",
        "my thoughts on the new design", "testing 123", "fix the login bug",
    ]
    text = random.choice(texts)
    templates = [
        "{pfx}open {app} and type {text}{sfx}",
        "{pfx}open {app} and write {text}{sfx}",
        "{pfx}launch {app} and type in {text}{sfx}",
        "{app} kholo aur likh do {text}{sfx}",
    ]
    t = random.choice(templates).format(
        pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES),
        app=app, text=text
    )
    tools = [{"name": "open_app_and_type", "arguments": {"app_name": app, "text_to_type": text}}]
    return t, tools


def make_pair_open_url():
    name, url = random.choice(WEBSITES)
    templates = [
        "{pfx}open {name}{sfx}",
        "{pfx}go to {name}{sfx}",
        "{pfx}take me to {name}{sfx}",
        "{pfx}open {url} in the browser{sfx}",
        "{pfx}navigate to {name}{sfx}",
        "{name} khol do{sfx}",
    ]
    t = random.choice(templates).format(
        pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES),
        name=name, url=url
    )
    tools = [{"name": "open_url", "arguments": {"url": url}}]
    return t, tools


def make_pair_search_web():
    query = random.choice(SEARCH_QUERIES)
    templates = [
        "{pfx}search for {query}{sfx}",
        "{pfx}Google {query}{sfx}",
        "{pfx}look up {query}{sfx}",
        "{pfx}find me {query}{sfx}",
        "{pfx}search {query} on the web{sfx}",
        "{query} search kar do{sfx}",
    ]
    t = random.choice(templates).format(
        pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES), query=query
    )
    tools = [{"name": "search_web", "arguments": {"query": query}}]
    return t, tools


def make_pair_run_computer_command():
    action_type, target = random.choice(COMPUTER_TARGETS)
    templates = [
        "{pfx}open {target}{sfx}",
        "{pfx}run {target}{sfx}",
        "{pfx}launch {target} for me{sfx}",
        "{pfx}press {target}{sfx}" if action_type == "press_key" else "{pfx}open {target}{sfx}",
        "{target} chalao{sfx}",
    ]
    t = random.choice(templates).format(
        pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES), target=target
    )
    tools = [{"name": "run_computer_command", "arguments": {"action_type": action_type, "target": target}}]
    return t, tools


def make_pair_analyze_screen():
    query = random.choice(SCREEN_QUERIES)
    templates = [
        "{pfx}{query}{sfx}",
        "{query}{sfx}",
    ]
    t = random.choice(templates).format(
        pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES), query=query
    )
    tools = [{"name": "analyze_screen_with_llava", "arguments": {"task_query": query}}]
    return t, tools


def make_pair_search_and_read_web():
    query = random.choice(SEARCH_QUERIES)
    templates = [
        "{pfx}research {query} and give me a summary{sfx}",
        "{pfx}find out about {query} from the internet{sfx}",
        "{pfx}read about {query} online and tell me{sfx}",
        "{pfx}scrape the web for {query}{sfx}",
    ]
    t = random.choice(templates).format(
        pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES), query=query
    )
    tools = [{"name": "search_and_read_web", "arguments": {"query": query}}]
    return t, tools


def make_pair_read_specific_url():
    name, url = random.choice(WEBSITES)
    templates = [
        "{pfx}read the content of {url}{sfx}",
        "{pfx}scrape {url} and tell me what it says{sfx}",
        "{pfx}fetch the text from {url}{sfx}",
        "{pfx}read {url} for me{sfx}",
    ]
    t = random.choice(templates).format(
        pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES), url=url
    )
    tools = [{"name": "read_specific_url", "arguments": {"url": url}}]
    return t, tools


def make_pair_write_and_run_script():
    instruction = random.choice(SCRIPT_INSTRUCTIONS)
    templates = [
        "{pfx}write {instruction}{sfx}",
        "{pfx}create {instruction} and run it{sfx}",
        "{pfx}code {instruction}{sfx}",
        "{pfx}make {instruction}{sfx}",
    ]
    t = random.choice(templates).format(
        pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES),
        instruction=instruction
    )
    tools = [{"name": "write_and_run_script", "arguments": {"instruction": instruction}}]
    return t, tools


def make_pair_get_system_diagnostics():
    query = random.choice(DIAGNOSTIC_QUERIES)
    t = f"{random.choice(PREFIXES)}{query}{random.choice(SUFFIXES)}"
    tools = [{"name": "get_system_diagnostics"}]
    return t, tools


def make_pair_media_control():
    action, value = random.choice(MEDIA_ACTIONS)
    if value is not None:
        templates = [
            "{pfx}set the volume to {value}{sfx}",
            "{pfx}volume {value}{sfx}",
            "{pfx}change volume to {value}{sfx}",
            "volume {value} kar do{sfx}",
        ]
        t = random.choice(templates).format(
            pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES), value=value
        )
        tools = [{"name": "media_control", "arguments": {"action": action, "value": value}}]
    else:
        templates = [
            "{pfx}{action} the music{sfx}",
            "{pfx}{action} playback{sfx}",
            "{pfx}{action} media{sfx}",
            "music {action} karo{sfx}",
        ]
        t = random.choice(templates).format(
            pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES), action=action
        )
        tools = [{"name": "media_control", "arguments": {"action": action}}]
    return t, tools


def make_pair_send_whatsapp():
    contact = make_contact()
    message = random.choice(MESSAGES)
    templates = [
        "{pfx}send a WhatsApp message to {contact} saying {message}{sfx}",
        "{pfx}text {contact} on WhatsApp that {message}{sfx}",
        "{pfx}WhatsApp {contact} and say {message}{sfx}",
        "{pfx}msg {contact} on whatsapp {message}{sfx}",
        "{contact} ko WhatsApp pe bhejo {message}{sfx}",
        "{pfx}tell {contact} on WhatsApp {message}{sfx}",
    ]
    t = random.choice(templates).format(
        pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES),
        contact=contact, message=message
    )
    tools = [{"name": "send_whatsapp_message", "arguments": {"contact_name": contact, "message": message}}]
    return t, tools


def make_pair_send_whatsapp_correction():
    contact1 = make_contact()
    contact2 = make_contact()
    # Ensure they are distinct
    while contact1 == contact2:
        contact2 = make_contact()
    message = random.choice(MESSAGES)
    templates = [
        "{pfx}Send a quick WhatsApp to {contact1}... actually make it {contact2}, telling them {message}{sfx}",
        "{pfx}text {contact1} on WhatsApp... wait no, msg {contact2} that {message}{sfx}",
        "{pfx}WhatsApp {contact1}... no wait, tell {contact2} on WhatsApp {message}{sfx}",
    ]
    t = random.choice(templates).format(
        pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES),
        contact1=contact1, contact2=contact2, message=message
    )
    tools = [{"name": "send_whatsapp_message", "arguments": {"contact_name": contact2, "message": message}}]
    return t, tools


def make_pair_set_timer():
    minutes = random.choice(TIMER_MINUTES)
    reminder = random.choice(REMINDERS)
    templates = [
        "{pfx}set a timer for {minutes} minutes to {reminder}{sfx}",
        "{pfx}remind me in {minutes} minutes to {reminder}{sfx}",
        "{pfx}set an alarm for {minutes} minutes and remind me to {reminder}{sfx}",
        "{minutes} minute ka timer laga do for {reminder}{sfx}",
    ]
    t = random.choice(templates).format(
        pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES),
        minutes=minutes, reminder=reminder
    )
    tools = [{"name": "set_timer", "arguments": {"minutes": minutes, "reminder_message": reminder}}]
    return t, tools


def make_pair_teach_new_skill():
    skill_name, description, instruction = random.choice(SKILL_TEMPLATES)
    templates = [
        "{pfx}teach yourself how to {description} by writing {instruction}{sfx}",
        "{pfx}learn a new skill called {skill_name} that can {description} using {instruction}{sfx}",
        "{pfx}create a permanent skill named {skill_name} to {description} with {instruction}{sfx}",
        "{pfx}I want you to learn to {description} by doing {instruction} and call it {skill_name}{sfx}",
        "{pfx}teach me a new skill called {skill_name} where you {description}{sfx}",
        "{pfx}plz teach new skill {skill_name} to {description}{sfx}"
    ]
    t = random.choice(templates).format(
        pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES),
        skill_name=skill_name, description=description, instruction=instruction
    )
    tools = [{"name": "teach_new_skill", "arguments": {
        "skill_name": skill_name, "description": description, "instruction": instruction
    }}]
    return t, tools


# ── Multi-tool compound generators ──────────────────────────────────

def make_compound_whatsapp_and_timer():
    contact = make_contact()
    message = random.choice(MESSAGES)
    minutes = random.choice(TIMER_MINUTES)
    reminder = random.choice(REMINDERS)
    t = f"{random.choice(PREFIXES)}send {contact} a WhatsApp saying {message} and set a {minutes} minute timer to {reminder}{random.choice(SUFFIXES)}"
    tools = [
        {"name": "send_whatsapp_message", "arguments": {"contact_name": contact, "message": message}},
        {"name": "set_timer", "arguments": {"minutes": minutes, "reminder_message": reminder}},
    ]
    return t, tools


def make_compound_search_and_message():
    contact = make_contact()
    query = random.choice(SEARCH_QUERIES)
    message = random.choice(MESSAGES)
    t = f"{random.choice(PREFIXES)}search for {query} and then text {contact} on WhatsApp saying {message}{random.choice(SUFFIXES)}"
    tools = [
        {"name": "search_web", "arguments": {"query": query}},
        {"name": "send_whatsapp_message", "arguments": {"contact_name": contact, "message": message}},
    ]
    return t, tools


def make_compound_smart_home_and_media():
    device = random.choice(DEVICES)
    room = random.choice(ROOMS)
    action = random.choice(DEVICE_ACTIONS)
    media_action, media_value = random.choice(MEDIA_ACTIONS)
    t = f"{random.choice(PREFIXES)}{action} the {device} in the {room} and {media_action} the music{random.choice(SUFFIXES)}"
    tools = [
        {"name": "handle_smart_home", "arguments": {"device": room, "action": action}},
        {"name": "media_control", "arguments": {"action": media_action}},
    ]
    return t, tools


def make_compound_time_and_whatsapp():
    contact = make_contact()
    message = random.choice(MESSAGES)
    t = f"{random.choice(PREFIXES)}check the time and then WhatsApp {contact} saying {message}{random.choice(SUFFIXES)}"
    tools = [
        {"name": "get_current_time"},
        {"name": "send_whatsapp_message", "arguments": {"contact_name": contact, "message": message}},
    ]
    return t, tools


def make_compound_url_and_type():
    name, url = random.choice(WEBSITES)
    app = random.choice(APPS)
    text = random.choice(["notes from the meeting", "my project ideas", "things to remember"])
    t = f"{random.choice(PREFIXES)}open {name} and also open {app} and type {text}{random.choice(SUFFIXES)}"
    tools = [
        {"name": "open_url", "arguments": {"url": url}},
        {"name": "open_app_and_type", "arguments": {"app_name": app, "text_to_type": text}},
    ]
    return t, tools


def make_compound_diagnostics_and_whatsapp():
    query = random.choice(DIAGNOSTIC_QUERIES)
    contact = make_contact()
    message = random.choice(MESSAGES)
    templates = [
        "{pfx}{query} and then send a whatsapp to {contact} saying {message}{sfx}",
        "{pfx}do this: {query}, then send a text to {contact} that {message}{sfx}",
        "{pfx}{query} & send whats app msg to {contact} saying {message}{sfx}",
    ]
    t = random.choice(templates).format(
        pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES),
        query=query, contact=contact, message=message
    )
    tools = [
        {"name": "get_system_diagnostics"},
        {"name": "send_whatsapp_message", "arguments": {"contact_name": contact, "message": message}},
    ]
    return t, tools


def make_compound_search_and_shell():
    query = random.choice(SEARCH_QUERIES)
    action_type, target = random.choice(COMPUTER_TARGETS)
    templates = [
        "{pfx}search the web for {query} then run a shell {target}{sfx}",
        "{pfx}google {query} and execute command {target}{sfx}",
    ]
    t = random.choice(templates).format(
        pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES),
        query=query, target=target
    )
    tools = [
        {"name": "search_web", "arguments": {"query": query}},
        {"name": "run_computer_command", "arguments": {"action_type": action_type, "target": target}},
    ]
    return t, tools


def make_compound_script_and_search():
    instruction = random.choice(SCRIPT_INSTRUCTIONS)
    query = random.choice(SEARCH_QUERIES)
    templates = [
        "{pfx}execute a terminal script to {instruction} and then pull up a google search for {query}{sfx}",
        "{pfx}run script to {instruction} and search {query} online{sfx}",
    ]
    t = random.choice(templates).format(
        pfx=random.choice(PREFIXES), sfx=random.choice(SUFFIXES),
        instruction=instruction, query=query
    )
    tools = [
        {"name": "write_and_run_script", "arguments": {"instruction": instruction}},
        {"name": "search_web", "arguments": {"query": query}},
    ]
    return t, tools


# ── All generators with weights ─────────────────────────────────────

GENERATORS = [
    (make_pair_get_current_time, 3),
    (make_pair_handle_smart_home, 5),
    (make_pair_route_to_deepseek, 4),
    (make_pair_open_app_and_type, 5),
    (make_pair_open_url, 5),
    (make_pair_search_web, 5),
    (make_pair_run_computer_command, 5),
    (make_pair_analyze_screen, 3),
    (make_pair_search_and_read_web, 4),
    (make_pair_read_specific_url, 3),
    (make_pair_write_and_run_script, 4),
    (make_pair_get_system_diagnostics, 4),
    (make_pair_media_control, 5),
    (make_pair_send_whatsapp, 8),
    (make_pair_send_whatsapp_correction, 4),
    (make_pair_set_timer, 5),
    (make_pair_teach_new_skill, 5),
    # Compound (multi-tool)
    (make_compound_whatsapp_and_timer, 4),
    (make_compound_search_and_message, 4),
    (make_compound_smart_home_and_media, 3),
    (make_compound_time_and_whatsapp, 3),
    (make_compound_url_and_type, 3),
    (make_compound_diagnostics_and_whatsapp, 5),
    (make_compound_search_and_shell, 5),
    (make_compound_script_and_search, 5)
]


def pick_generator():
    funcs, weights = zip(*GENERATORS)
    return random.choices(funcs, weights=weights, k=1)[0]


def generate_fill(target_count=1500):
    dataset_file = os.path.join(os.path.dirname(__file__), "aether_orchestrator_dataset.jsonl")

    existing = 0
    if os.path.exists(dataset_file):
        with open(dataset_file, "r", encoding="utf-8") as f:
            existing = sum(1 for _ in f)

    needed = target_count - existing
    if needed <= 0:
        print(f"Already have {existing} pairs (target={target_count}). Nothing to do.")
        return

    print(f"Existing: {existing}. Generating {needed} template-based pairs to reach {target_count}...")

    system_msg = "You are a tool-calling orchestrator. Output ONLY a raw JSON array of tool calls. No text, no markdown, no explanation."
    generated = 0

    with open(dataset_file, "a", encoding="utf-8") as f:
        while generated < needed:
            gen_func = pick_generator()
            user_prompt, tool_calls = gen_func()

            # Clean up whitespace
            user_prompt = " ".join(user_prompt.split())

            # Verify substring alignment for every argument (skip categorical ones like action_type, action)
            valid = True
            for tool in tool_calls:
                for key, val in tool.get("arguments", {}).items():
                    if key in ["action_type", "action"]:
                        continue
                    str_val = str(val)
                    if str_val not in user_prompt:
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                continue

            expected_json = json.dumps(tool_calls)

            row = {
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": expected_json},
                ]
            }
            f.write(json.dumps(row) + "\n")
            generated += 1

            if generated % 500 == 0:
                print(f"  Generated {generated}/{needed}...")

    print(f"Done! Added {generated} pairs. Total dataset: {existing + generated}.")


if __name__ == "__main__":
    dataset_file = os.path.join(os.path.dirname(__file__), "aether_orchestrator_dataset.jsonl")
    if os.path.exists(dataset_file):
        print(f"Removing old dataset file {dataset_file} to ensure a fresh, balanced generation.")
        os.remove(dataset_file)
    generate_fill(target_count=5000)
