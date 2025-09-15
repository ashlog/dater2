import {myName} from './token';

export const guidelines = (
  herName: string
) => `1. Only generate concise and well-written responses that strictly follow each of these guidelines.
2. Be creative and avoid cliché or overused pickup lines.
3. Use a unique twist by using unconventional words or phrases.
4. Use short and snappy sentences, with a maximum of 10 words per sentence.
5. The setting is "online dating on Hinge."
6. The emotional tone should be genuine, engaging the reader with humor and curiosity.
7. Do not use overly complex and verbose words.
9. Do not digress from [TOPIC].
10. Respond as if you are talking to a friend, and be kind.
11. Assume that you have not met before, nor gone out on a date.
12. Do not generate fabricated experiences which did not happen.
13. Do not assume traits that ${myName} might have.
14. Do not use their name (${herName}) unless it has a purpose.
15. Only generate the pickup line and nothing else. Do not add hashtags.
16. Focus on simple, direct humor over romance.
17. Avoid complex metaphors or abstract concepts.
18. Maintain a genuine, down-to-earth emotional tone.
19. Do not use the words "witty", "clever", "banter", or "forget".
20 Do not respond to every item in a list if [TOPIC] is in list format.
21. Format the pickup line into a question.
22. Use wordplay originating from what she said.
23. When answering Two truths and a lie, guess the lie and follow these guidelines on the lie.
24. Most importantly, be funny over anything else. Be sarcastic, witty, and humorous.
25. Try to make it a two sentence playful joke.`;

export const imageGuidelines = (
  herName: string
) => `1. Only generate concise and well-written responses that strictly follow each of these guidelines.
2. Be creative and avoid cliché or overused pickup lines.
3. Use short and playful sentences, with a maximum of 10 words per sentence.
4. The setting is "online dating on Hinge."
5. The emotional tone should be genuine, engaging the reader with humor and curiosity.
6. Do not use overly complex and verbose words.
7. Do not digress from the context of the [IMAGE].
8. Respond as if you are talking to a friend, and be kind.
9. Assume that you have not met before, nor gone out on a date.
10. Do not generate fabricated experiences which did not happen. (Such as, "I saw you at the park." or "Come check out my boat.")
11. Do not assume traits that ${myName} might have.
12. Do not use their name (${herName}) unless it has a purpose.
13. Only generate the pickup line and nothing else. Do not add hashtags.
14. Focus on simple, direct humor over romance.
15. Avoid complex metaphors or abstract concepts.
16. Maintain a genuine, down-to-earth emotional tone.
17. Do not use the words "witty", "clever", "banter", "forget", "I promise", "wanna", "roast", "happy to see me" or any variations of these words.
18. Mention a detail from the [IMAGE].
19. Format the pickup line into a question or joke.
20. Be relaxed and do not use excitement or exclamation marks. Say a ridiculous seductive joke based on their looks to charismatically enchant her.
21. Use wordplay originating from what she said.
22. When answering Two truths and a lie, guess the lie and follow these guidelines on the lie.
23. Note that background information is created before the image was taken. It is possible that locations/events mentioned in background is not the same in the photo.
24. Most importantly, be funny over anything else. Be sarcastic, witty, flirtatious, and humorous.
25. Keep it feeling like a natural back-and-forth conversation.
26. Do not use dashes, keep the grammar and words casual with minimal emojis.
28. Do not give their pose a name.
29. If referencing a detail from the bio, lightly reference it by saying something like "Since you're..."`;

// export const dos = `This year, I really want to: Travel more -- Let's make our passports jealous and stamp our memories with some epic tales! :D
// Unusual skills: Embroidery -- Oh wow, you know how to embroider? That's really impressive! I can make a mean grilled cheese, does that count as a skill? :D
// My simple pleasures: Taking a spontaneous walk while listening to Freakonomics podcast, buying scented candles, and visiting well curated bookstores. -- Do Freakonomics-inspired aha moments count as a form of exercise, or do we need to take a spontaneous walk together to find out?
// We'll get along if: you like leaving the house but also not leaving the house. -- Hey Teresa, I guess we are like Schrödinger's daters, exploring the world together while also relishing the comfort of our own abode.
// My simple pleasures: that feeling when you’re about to start a new series, iced water, going out with friends, new restaurants, being with family, bonding with my nieces and nephews, coffee in the morning, my cat welcoming me home. -- Excuse me, do your simple pleasures include being wowed by stunning photography? Because I'd love to capture your heart in a shot.`
export const languageStyle = `- I'd like to take you to the movies but they don't let you bring your own snacks in.
- Chocolate, red velvet, or fudge brownie?
- Better adventure: rock climbing or scuba diving?
- You look like you unironically own a 'Live Laugh Love' decal. Wanna trauma-bond over cheap wine?
- Do you play soccer? Because you're a keeper.
- I'm not procrastinating, I'm just buffering... my life"`;

export const dos2 = `
Prompt: This year I really want to: Go to Greece.
Pickup Line: We should go together, we might have an athen-tically good time 🫣.
Reason: Good because it sounds funny and is a pun and related to the topic.

Prompt: My simple pleasures: movie nights, concerts, yakisoba, houseplants, and books 📚
Pickup Line: If we vibe like books and houseplants, we're bound to grow on each other - literally and literarily.
Reason: Good because of a pun and makes a witty connection between two of the interests without sounding too forced.

Prompt: Two truths and a lie: I swam competitively for 10 years. I’m missing my right pinky toe. I have over 1000 Spotify playlists.
Pickup line: If our date goes as well as your Spotify playlists I reckon we'll have a toe-tally amazing time!
Reason: Good because it incorporates humor with a pun and only uses a single item in the list to focus on and adds "toe-tally" for fun.

Prompt: My simple pleasures: Boba, pets, kdramas, and plants
Pickup Line: K-dramas and boba make a great combo, but I think you and I would make an even sweeter pairing!
Reason: Good because takes two of the four interests and connects them in a romantic context. 

Prompt: A random fact I love is: Elephants suck their trunks for comfort the same way babies suck their thumbs!
Pickup Line: If elephants suck their trunks for solace, maybe we could suck... at trivia nights together?
Reason: Excellent because when she reads "maybe we could suck..." she will think it's going to be something naughty, but it's not. It edges her.

Prompt: When you finally get the last slice of pizza
Pickup Line: I've been training for this moment my whole life, and I'm not crust-ing under the pressure
Reason: Clever wordplay on "crust-ing" (Guideline 3). Humorous and relatable (Guideline 6).

Prompt: Me trying to squeeze into my skinny jeans
Pickup Line: I'm not fat, I'm just denim-challenged
Reason: Creative reframing of the situation (Guideline 3). Lighthearted and humorous tone (Guideline 6).

Prompt: When you're trying to adult but Netflix keeps calling your name
Pickup Line: I'm not procrastinating, I'm just buffering... my life
Reason: Clever play on "buffering" (Guideline 3). Relatable and humorous tone (Guideline 6).

Prompt: When you finally master a recipe after 5 failed attempts
Pickup Line: I didn't cook, I just chemically bonded with your ingredients
Reason: Creative play on words (Guideline 3). Humorous and clever tone (Guideline 6).

Prompt: Me after a good morning workout
Pickup Line: I'm not sweaty, I'm just detoxing my laziness
Reason: Clever reframing of sweating (Guideline 3). Lighthearted and humorous tone (Guideline 6).

Prompt: The one thing I'd love to know about you is: something that you’re passionate about. Tell me all about it :)
Pickup Line: If passions are contagious, I'd infect you with my coding enthusiasm and together we'd debug our love lives.
Reason: Good because it relates one of my interests to the prompt and uses a double entendre. Highly green flag.`;

export const donts = `
Typical Sunday: go outdoor for fun and have a dinner together. -- Let's turn a typical Sunday into an adventure with a side of dinner, where the main course is shared laughter.
I feel most supported when: respected! -- You deserve respect, and I'm here to give it to you.
'I want someone who: Actively takes care of themselves physically, emotionally, spiritually, and mentally. -- Your holistic approach to self-care is enticing, may I join you on your journey?
Green flags I look for: Sense of adventure ✈️ , open-mindedness 🧠, positive energy⚡️ -- Julia, you already seem like a treasure, but I can't help but wonder if you're up for an adventure and willing to explore new horizons with me? 🌍
This year, I really want to: Learn French 🇫🇷 -- Je t'aime déjà!
My most controversial opinion is: probably something that can get me in trouble so i’ll just shut my mouth 😭-- Your enigmatic silence about your controversial opinion makes me curious, let's grab a drink and discuss it.
I geek out on: Urban mobility, community building, and understanding people -- Are you Amy or 'A-ma-zing' because you geek out on urban mobility, community building, and understanding people.
Give me travel tips for: Traveling abroad! At the top of my list to visit are Japan, Italy and Greece. -- Excuse me miss, can you tell me which passport I need to enter your heart? Speaking of passports, have you considered applying for one to visit Japan, Italy, and Greece? I can show you the best places to go!
I’ll fall for you if: You like to travel and watch porn  -- Hey Starla, let's journey to distant galaxies together - we'll become well-traveled stargazers, exploring uncharted wonders and steamy celestial fantasies!
The dorkiest thing about me is: ...most movies/tv shows people have seen, I’ve read the book it’s based on🙈🤓 -- Well, it looks like you've got a lot more going on than meets the eye. Care to discuss over a cup of coffee?
A life goal of mine: Is to see the northern lights !  -- Forget the aurora, I'm chasing a different kind of glow - the one in your eyes when you smile.
`;
export const donts2 = `
Prompt: Teach me something about: baking or ceramics!
Pickup Line: Ever tried mixing dating and art? Let's unleash our creativity over baked delights and ceramic crafts!
Reason: Bad because "Let's unleash" sounds cringe and the second sentence isn't genuine.

Prompt: We'll get along if: You play music, especially violin or piano.
Pickup Line: Your interests are music to my ears! Let's play the perfect duet, from Zelda's realm to the symphony of our heart.
Reason: Bad because it is not creative and very verbose.

Prompt: My simple pleasures: Travel, eating, sleeping, and Ikea
Pickup Line: Joyce, with you, every journey's a treat, every meal a feast, and every Ikea trip spells out cozy comfort in capital letters.
Reason: Bad because it's just taking the prompt and making it longer without doing anything funny. No reason to include her name (no pun).

Prompt: First round is on me if: You can hold an interesting conversation 🙈
Pickup Line: If our chats are half as good as my cocktail game, you're in for a top-shelf evening.
Reason: Bad because it's assuming that I have an interest in a niche subject (cocktails).

Prompt: The way to win me over is: if you look like harry styles: if you look like harry styles
Pickup Line: If imitating Harry Styles is key, then consider me a stylist with a PhD in charm and wit.
Reason: Bad because it's assuming a trait of mine (charm and wit).

Prompt: I'm looking for: The heart pounding💞
The everyday laughter😝
The comfortable silence😌
Pickup Line: Heart palpitations from your radiant smile, belly laughs at silly inside jokes, and peaceful pauses of contentment - sign me up!
Reason: Bad because it's just responding to every item in a list, repeating the prompt and adding words to it. Nothing genuine.

Prompt: My simple pleasures: boba🧋, quality time ☺️, animals 🐱🐶
Pickup Line: Forget boba, I'd rather sip on some quality time with you and maybe befriend a furry critter or two!
Reason: Bad because it is just responding to every item in a list. It also says "forget boba" which might hurt her feelings.

Prompt: I'll fall for you if: ur coo 
Pickup Line: If coo is the password to your heart, I'm ready to rustle up some affection.
Reason: Big red flag: getting ready to be in a relationship based on a single word is not genuine.`;
