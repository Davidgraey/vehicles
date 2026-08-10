import pygame


class InputController:
    def __init__(self):
        self._commands = {'turn': 0.0, 'accelerate': 0.0}
        self._keys_pressed = set()

    def process_events(self):
        """Call once per frame with pygame.event.get()"""
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                return False  # signal shutdown

            if (event.type == pygame.KEYDOWN):
                self._keys_pressed.add(event.key)

            elif (event.type == pygame.KEYUP):
                self._keys_pressed.discard(event.key)

        # Map keys to normalized deltas (frame-independent scaling handled in loop)
        if pygame.K_UP in self._keys_pressed:
            self._commands['accelerate'] = 1.0
        else:
            self._commands['accelerate'] = 0.0

        if pygame.K_DOWN in self._keys_pressed:
            self._commands['accelerate'] = -1.0  # brake/reverse

        if pygame.K_LEFT in self._keys_pressed:
            self._commands['turn'] = 0.1
        elif pygame.K_RIGHT in self._keys_pressed:
            self._commands['turn'] = -0.1
        else:
            self._commands['turn'] = 0.0

    def get_commands(self) -> dict:
        return self._commands.copy()

    def reset(self):
        self._keys_pressed.clear()
        self._commands = {'turn': 0.0, 'accelerate': 0.0}
