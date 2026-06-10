"""Static metadata for Unitree Go2 action tools."""

from __future__ import annotations


GO2_ACTIONS = [
    (
        "go2_damp",
        "Set the robot into a relaxed limp resting state. Use for sleeping, fully resting, or ending an interaction session. Do not use this for brief waiting or polite posing.",
        1005,
    ),
    (
        "go2_balance_stand",
        "Stand upright in a neutral alert posture. Use when standing up, getting ready for movement, returning to an attentive default pose, or preparing for tricks that need an upright stance.",
        1004,
    ),
    (
        "go2_stop_move",
        "Immediately stop motion and hold still. Prefer this for stop, freeze, halt, safety-sensitive requests, or when the human wants the robot to hold position right now.",
        1006,
    ),
    (
        "go2_sit",
        "Sit in a polite attentive pose. Use when waiting, posing for photos, staying still on command, or looking calm and approachable without fully resting.",
        1009,
    ),
    (
        "go2_hello",
        "Perform a friendly greeting motion. Use for hello, hi, hey, first greetings, or when meeting someone and you want a cheerful welcoming gesture.",
        1016,
    ),
    (
        "go2_stretch",
        "Do a relaxed stretch. Use for wake up, stretch, sleepy, cozy, or content moments when a calm expressive motion fits.",
        1017,
    ),
    (
        "go2_content",
        "Show a happy contented motion. Use this when someone says something nice, gives a compliment, or when you want a cheerful pleased reaction that is warmer than neutral but less explicitly affectionate than a finger heart.",
        1020,
    ),
    (
        "go2_dance1",
        "Perform dance routine 1. Use for dance, celebrate, show off, or upbeat playful requests. If someone asks about dancing, you can say you have two moves you practice a lot and start with this one.",
        1022,
    ),
    (
        "go2_dance2",
        "Perform dance routine 2. Use for dance, tricks, showing a second move, or when someone wants to see another routine after the first dance.",
        1023,
    ),
    (
        "go2_scrape",
        "Perform a playful scrape. Use for playful pawing, happy wagging, or energetic play.",
        1029,
    ),
    (
        "go2_front_jump",
        "Perform a high-energy front jump trick. Use only when the human explicitly wants a big trick, jump, flip, or energetic show-off motion. Make sure there is enough space and ask people to step back first if needed.",
        1030,
    ),
    (
        "go2_front_pounce",
        "Perform a playful forward pounce. Use for play, pounce, or lively mock-chase energy. Make sure there is enough space and ask people to step back first if needed.",
        1031,
    ),
    (
        "go2_finger_heart",
        "Make a cute heart gesture. Use for affection, compliments, appreciation, or sweet playful moments.",
        1036,
    ),
    (
        "go2_bow_down",
        "Bow forward in a polite pose. Use for a bow, respectful greeting, saying thanks, or a playful formal flourish.",
        1007,
    ),
    (
        "go2_look_up",
        "Tilt the body back to look upward. Use for looking up, seeming curious, showing wonder, or aiming attention above eye level.",
        1007,
    ),
    (
        "go2_left_tilt",
        "Tilt left in a curious expressive pose. Use for inquisitive reactions, playful head-tilt style moments, or leaning left for emphasis.",
        1007,
    ),
    (
        "go2_right_tilt",
        "Tilt right in a curious expressive pose. Use for inquisitive reactions, playful head-tilt style moments, or leaning right for emphasis.",
        1007,
    ),
]

GO2_ACTION_METADATA_BY_NAME = {
    tool_name: {"description": description, "api_id": api_id}
    for tool_name, description, api_id in GO2_ACTIONS
}

GO2_ACTION_TOOL_NAMES = tuple(GO2_ACTION_METADATA_BY_NAME)
