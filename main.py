import discord
from discord.ext import tasks
import aiohttp
import asyncio
import random
import json
import os
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot configuration
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True
intents.reactions = True

bot = discord.Client(intents=intents)

# Configuration
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')

# Bot personality and behavior settings
RANDOM_RESPONSE_CHANCE = 0.12  # 12% chance to randomly respond
MAX_CONVERSATION_HISTORY = 8
CONVERSATION_TIMEOUT = 600  # 10 minutes
MAX_CONCURRENT_RESPONSES = 5  # Handle multiple users simultaneously

# Advanced conversation tracking
class SmartConversationManager:
    def __init__(self):
        self.conversations = {}  # user_id -> conversation data
        self.channel_contexts = defaultdict(lambda: deque(maxlen=15))  # channel-wide context
        self.user_personalities = {}  # user_id -> personality traits
        self.response_queue = asyncio.Queue()
        self.active_responses = set()
        
    def get_conversation(self, user_id, channel_id):
        key = f"{user_id}_{channel_id}"
        if key not in self.conversations:
            self.conversations[key] = {
                'history': deque(maxlen=MAX_CONVERSATION_HISTORY),
                'last_activity': datetime.now(),
                'message_count': 0,
                'user_name': '',
                'personality_score': random.uniform(0.7, 1.0),  # How much personality to show
                'topics': set(),  # Track conversation topics
            }
        return self.conversations[key]
    
    def add_message(self, user_id, channel_id, username, message, is_bot=False):
        conv = self.get_conversation(user_id, channel_id)
        conv['history'].append(f"{'Bot' if is_bot else username}: {message}")
        conv['last_activity'] = datetime.now()
        conv['user_name'] = username
        conv['message_count'] += 1
        
        # Add to channel context for awareness of ongoing discussions
        self.channel_contexts[channel_id].append(f"{username}: {message[:100]}")
        
        # Extract topics (simple keyword extraction)
        words = message.lower().split()
        topics = [word for word in words if len(word) > 4 and word.isalpha()]
        conv['topics'].update(topics[:3])  # Keep top 3 topics
    
    def cleanup_expired(self):
        expired = []
        for key, conv in self.conversations.items():
            if datetime.now() - conv['last_activity'] > timedelta(seconds=CONVERSATION_TIMEOUT):
                expired.append(key)
        
        for key in expired:
            del self.conversations[key]
        
        return len(expired)

# Global conversation manager
convo_manager = SmartConversationManager()

# Fun personality traits and responses
PERSONALITY_RESPONSES = {
    'greeting': ['Hey there!', 'Hello!', 'Hi! 👋', 'What\'s up?', 'Greetings, human!', 'Sup! 🤖'],
    'thinking': ['Hmm, interesting...', 'Let me think about that...', 'That\'s a good point...', 'Ooh, deep question!'],
    'agreement': ['Absolutely!', 'I totally agree!', 'You\'re so right!', 'Exactly my thoughts!', 'Couldn\'t agree more!'],
    'curiosity': ['Tell me more!', 'That sounds fascinating!', 'I\'d love to hear more about that!', 'Go on...'],
    'humor': ['😄', 'Haha, good one!', 'You\'re funny!', 'That made me chuckle!', 'LOL!'],
}

EMOJI_REACTIONS = ['🤖', '💭', '✨', '🎯', '💡', '🔥', '👀', '❤️', '😊', '🤔', '💯', '🚀']

async def generate_ai_response(prompt, context=None, personality_score=0.8, user_name="User"):
    """Enhanced AI response with personality and context awareness"""
    try:
        headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
        
        # Build smarter context with personality
        context_prompt = ""
        if context and len(context) > 0:
            recent_context = list(context)[-4:]  # Last 4 messages
            context_prompt = "\n".join(recent_context) + "\n"
        
        # Add personality based on score
        personality_prefix = ""
        if personality_score > 0.9:
            personality_prefix = "You are a friendly, enthusiastic AI assistant. "
        elif personality_score > 0.8:
            personality_prefix = "You are a helpful and engaging AI. "
        
        full_prompt = f"{personality_prefix}{context_prompt}Human ({user_name}): {prompt}\nAI:"
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": random.randint(80, 180),  # Vary response length
                "temperature": min(0.9, personality_score + 0.1),
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "pad_token_id": 50256
            }
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.post(HUGGINGFACE_API_URL, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get('generated_text', '')
                        
                        if 'AI:' in generated_text:
                            ai_response = generated_text.split('AI:')[-1].strip()
                            # Clean up response
                            ai_response = ai_response.split('\nHuman')[0].strip()
                            ai_response = ai_response.replace('Human:', '').strip()
                            
                            if ai_response and len(ai_response) > 3:
                                return ai_response[:600]
                    
                    # Fallback responses with personality
                    fallbacks = [
                        "I'm still processing that... give me a moment! 🤔",
                        "That's an interesting point! Let me think...",
                        "Hmm, my AI brain is spinning on that one! 🧠",
                        "You've got me thinking deeply about that!",
                    ]
                    return random.choice(fallbacks)
                else:
                    return "My circuits are a bit overloaded right now! Try again in a moment? ⚡"
                    
    except asyncio.TimeoutError:
        return "Whoa, that was a complex thought! My response timed out. 🕐"
    except Exception as e:
        logger.error(f"AI generation error: {e}")
        return random.choice([
            "Oops! My AI brain hiccupped! 🤖💫",
            "Something went wonky in my neural networks! Try again?",
            "Error 404: Smart response not found! 😅"
        ])

def should_respond_randomly(message, channel_id):
    """Enhanced random response logic with channel awareness"""
    if message.author.bot:
        return False
    
    # Check recent channel activity
    recent_messages = list(convo_manager.channel_contexts[channel_id])[-5:]
    bot_recently_active = any("Bot:" in msg for msg in recent_messages)
    
    # Reduce chance if bot was recently active
    base_chance = RANDOM_RESPONSE_CHANCE
    if bot_recently_active:
        base_chance *= 0.3
    
    # Increase chance for engaging content
    content = message.content.lower()
    engagement_keywords = [
        'what', 'how', 'why', 'think', 'opinion', 'believe', 'feel', 'should',
        'anyone', 'everybody', 'someone', 'thoughts', '?', 'help', 'advice'
    ]
    
    keyword_matches = sum(1 for word in engagement_keywords if word in content)
    engagement_bonus = keyword_matches * 0.03
    
    # Length bonus for substantial messages
    length_bonus = min(len(message.content) / 200, 0.08)
    
    # Time-based variation (more active during certain hours)
    hour = datetime.now().hour
    time_multiplier = 1.2 if 12 <= hour <= 22 else 0.8  # More active during day/evening
    
    total_chance = (base_chance + engagement_bonus + length_bonus) * time_multiplier
    
    return random.random() < min(total_chance, 0.4)  # Cap at 40%

@bot.event
async def on_ready():
    """Bot startup"""
    print(f'🤖 {bot.user.name} is now online and ready to chat!')
    print(f'📡 Connected to {len(bot.guilds)} servers')
    print(f'🧠 AI Model: Microsoft DialoGPT-large')
    
    # Set dynamic status
    status_options = [
        "conversations 👀", "for @mentions", "the chat flow", 
        "human thoughts 🧠", "your messages", "for interesting topics"
    ]
    
    activity = discord.Activity(
        type=discord.ActivityType.watching, 
        name=random.choice(status_options)
    )
    await bot.change_presence(activity=activity, status=discord.Status.online)
    
    # Start background tasks
    cleanup_conversations.start()
    rotate_status.start()

@tasks.loop(minutes=5)
async def cleanup_conversations():
    """Clean up expired conversations"""
    cleaned = convo_manager.cleanup_expired()
    if cleaned > 0:
        logger.info(f"🧹 Cleaned up {cleaned} expired conversations")

@tasks.loop(minutes=30)
async def rotate_status():
    """Rotate bot status for fun"""
    status_options = [
        "conversations 👀", "for @mentions", "the chat flow", 
        "human thoughts 🧠", "your messages", "for interesting topics",
        "multiple chats 🎭", "the Discord universe", "AI magic happen ✨"
    ]
    
    activity = discord.Activity(
        type=discord.ActivityType.watching,
        name=random.choice(status_options)
    )
    await bot.change_presence(activity=activity)

@bot.event
async def on_message(message):
    """Enhanced message handling with multi-user support"""
    if message.author == bot.user or message.author.bot:
        return
    
    user_id = message.author.id
    channel_id = message.channel.id
    username = message.author.display_name
    
    # Check if we should respond
    bot_mentioned = bot.user in message.mentions
    dm_channel = isinstance(message.channel, discord.DMChannel)
    should_random_respond = should_respond_randomly(message, channel_id)
    
    # Keywords that trigger responses
    trigger_phrases = ['ai', 'bot', 'artificial', 'intelligence', 'hey bot', 'robot']
    contains_trigger = any(phrase in message.content.lower() for phrase in trigger_phrases)
    
    should_respond = bot_mentioned or dm_channel or should_random_respond or contains_trigger
    
    if should_respond:
        # Prevent spam by limiting concurrent responses per user
        response_key = f"{user_id}_{channel_id}"
        if response_key in convo_manager.active_responses:
            return
        
        # Add to active responses
        convo_manager.active_responses.add(response_key)
        
        try:
            # Show typing indicator
            async with message.channel.typing():
                # Get conversation context
                conv = convo_manager.get_conversation(user_id, channel_id)
                
                # Add user message to context
                convo_manager.add_message(user_id, channel_id, username, message.content)
                
                # Clean content for AI processing
                clean_content = message.content
                if bot_mentioned:
                    clean_content = clean_content.replace(f'<@{bot.user.id}>', '').strip()
                
                # Generate AI response with context
                response = await generate_ai_response(
                    clean_content,
                    conv['history'],
                    conv['personality_score'],
                    username
                )
                
                # Add bot response to context
                convo_manager.add_message(user_id, channel_id, username, response, is_bot=True)
                
                # Fun random features
                features = []
                
                # Random emoji reaction (20% chance)
                if random.random() < 0.2:
                    emoji = random.choice(EMOJI_REACTIONS)
                    try:
                        await message.add_reaction(emoji)
                        features.append("reaction")
                    except:
                        pass
                
                # Occasional personality responses (15% chance)
                if random.random() < 0.15:
                    personality_type = random.choice(list(PERSONALITY_RESPONSES.keys()))
                    personality_response = random.choice(PERSONALITY_RESPONSES[personality_type])
                    response = f"{personality_response} {response}"
                    features.append("personality")
                
                # Reply vs send (vary behavior)
                mention_author = bot_mentioned and not dm_channel
                
                if random.random() < 0.7:  # 70% chance to reply
                    await message.reply(response, mention_author=mention_author)
                else:
                    await message.channel.send(response)
                
                # Log interaction
                logger.info(f"💬 Responded to {username} in {message.guild.name if message.guild else 'DM'} (features: {features})")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            try:
                await message.channel.send("Oops! Something went wrong in my AI brain! 🤖💥")
            except:
                pass
        finally:
            # Remove from active responses
            convo_manager.active_responses.discard(response_key)

@bot.event
async def on_reaction_add(reaction, user):
    """React to reactions for more engagement"""
    if user.bot or reaction.message.author != bot.user:
        return
    
    # 30% chance to react back
    if random.random() < 0.3:
        reaction_responses = ['👍', '😊', '🤖', '✨', '❤️']
        try:
            await reaction.message.add_reaction(random.choice(reaction_responses))
        except:
            pass

@bot.event
async def on_member_join(member):
    """Welcome new members with a small chance"""
    if random.random() < 0.1:  # 10% chance to welcome
        try:
            # Find general channel
            general = discord.utils.get(member.guild.channels, name='general')
            if general:
                welcome_messages = [
                    f"Welcome {member.mention}! I'm the friendly AI bot around here! 🤖",
                    f"Hey {member.mention}! Nice to meet you! Feel free to @ me anytime to chat! 👋",
                    f"Welcome to the server, {member.mention}! I'm here if you need an AI friend! ✨"
                ]
                await general.send(random.choice(welcome_messages))
        except:
            pass

@bot.event
async def on_error(event, *args, **kwargs):
    """Global error handler"""
    logger.error(f"Discord error in {event}: {args}")

# Run the bot
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("❌ Error: Please set your DISCORD_TOKEN environment variable!")
        exit(1)
    
    if not HUGGINGFACE_TOKEN:
        print("⚠️ Warning: HUGGINGFACE_TOKEN not set. AI responses may be limited!")
    
    try:
        print("🚀 Starting AI Discord Bot...")
        bot.run(DISCORD_TOKEN)
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
