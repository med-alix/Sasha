UNITS = {"chl": "mg/m^3", "cdom": "m^-1", "tsm": "g/m^3", "alpha_m": "ratio"}


def pretty_print_parameters(selected_params):
    # ANSI escape codes for colors and bold text
    RED_BOLD = "\033[91m\033[1m"
    GREEN_BOLD = "\033[92m\033[1m"
    YELLOW_BOLD = "\033[93m\033[1m"
    BLUE_BOLD = "\033[94m\033[1m"
    RESET = "\033[0m"

    print(
        f"{RED_BOLD}Selected Parametrization:{RESET} {GREEN_BOLD}{selected_params['label']}{RESET}"
    )

    print(f"{YELLOW_BOLD}Parameters:{RESET}")
    for param, value in selected_params.items():
        if param != "label":  # Skip the label entry
            unit = UNITS.get(param, "")
            print(f"  {BLUE_BOLD}{param}{RESET}: {GREEN_BOLD}{value} {unit}{RESET}")




