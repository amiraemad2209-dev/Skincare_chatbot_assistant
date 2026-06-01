from resolver.models import ShortcutDefinition


SHORTCUT_REGISTRY = {

    "/routine": ShortcutDefinition(
        trigger="/routine",

        expansion="""
        Build a professional dermatologist skincare routine.

        Include:
        - cleanser
        - treatment
        - moisturizer
        - sunscreen

        Consider the following user request:
        """,

        temperature=0.3,
        max_tokens=700,
        use_memory=True
    ),


    "/Keep Safe": ShortcutDefinition(
        trigger="/Keep Safe",

        expansion="""
        Analyze skincare safety and allergy concerns.

        Mention possible irritants and patch testing advice.

        User request:
        """,

        temperature=0.2,
        max_tokens=600,
        use_memory=True
    )
}