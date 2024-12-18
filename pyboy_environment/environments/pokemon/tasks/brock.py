from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc


class PokemonBrock(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:

        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            #WindowEvent.PRESS_BUTTON_START,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            #WindowEvent.RELEASE_BUTTON_START,
        ]

        super().__init__(
            act_freq=act_freq, #original is 24
            task="brock",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            valid_actions=valid_actions,
            release_button=release_button,
            headless=headless,
        )
    
        self.same_loc = 0
        self.turn_unreward = 0 
        self.y_reward = 0
        self.prev_turn = -1
        self.prev_loc = set()
        self.prev_spot = set()
        self.no_move = 0
        self.prev_enemy_hp = 0
        self.prev_menu = 0
        self.prev_item_sel = 0
        self.button_pressed = 0
        self.prev_button_pressed = 0
        self.no_attack = 0
        self.in_battle = False
        self.battle_win = 0
        self.ran_away = 0

    def reset(self) -> np.ndarray:
        self.steps = 0

        with open(self.init_path, "rb") as f:
            self.pyboy.load_state(f)

        self.prior_game_stats = self._generate_game_stats()

        self.same_loc = 0
        self.turn_unreward = 0 
        self.y_reward = 0
        self.prev_turn = -1
        self.prev_loc = set()
        self.prev_spot = set()
        self.no_move = 0
        self.prev_enemy_hp = 0
        self.prev_menu = 0
        self.prev_item_sel = 0
        self.button_pressed = 0
        self.prev_button_pressed = 0
        self.no_attack = 0
        self.in_battle = False
        self.battle_win = 0
        self.ran_away = 0

        return self._get_state()

    def _run_action_on_emulator(self, action_array: np.ndarray) -> None:
        action = action_array[0]
        action = min(action, 0.99)

        # Continuous Action is a float between 0 - 1 from Value based methods
        # We need to convert this to an action that the emulator can understand
        bins = np.linspace(0, 1, len(self.valid_actions) + 1)
        button = np.digitize(action, bins) - 1
        self.button_pressed = button
        # Push the button for a few frames
        self.pyboy.send_input(self.valid_actions[button])

        for _ in range(self.act_freq):
            self.pyboy.tick()

        # Release the button
        self.pyboy.send_input(self.release_button[button])

    def _read_items(self) -> dict:
        total_items = self._read_m(0xD31D)
        if total_items == 0:
            return {}
        
        addr = 0xD31E #Start of item list
        item = {}

        for i in range(total_items):
            item_id = self._read_m(addr + 2 * 1)
            item_count = self._read_m(addr+ 2*1 + 1)
            item[item_id] = item_count
        
        return item
    
    def _get_pokeball_count(self, items) -> int:
        total_count = 0
        for item_id, count in items.items():
            if item_id in range(0x0, 0x5):
                total_count += count
        
        return total_count

    def _movement_reward(self, new_state: dict[str, any]) -> int:
        reward = 0
        if self.in_battle:
            return reward

        if (new_state["location"]["x"], new_state["location"]["y"]) != (self.prior_game_stats["location"]["x"], self.prior_game_stats["location"]["y"]) and self._read_m(0xD057) == 0:
            #new coordinate unlocked
            if (new_state["location"]["x"], new_state["location"]["y"]) not in self.prev_spot:
                # if self._grass_reward(new_state) and not (self.prior_game_stats["location"]["map"] == 'ROUTE_1,' and new_state["location"]["map"] == "PALLET_TOWN,") :
                #     reward += 1.5
                # else:
                reward += 1
                self.prev_spot.add((new_state["location"]["x"], new_state["location"]["y"]))
                if new_state["location"]["map"] == "PALLET_TOWN,":
                    reward += 2
                elif new_state["location"]["map"] == "ROUTE_1,":
                    reward += 3
                elif new_state["location"]["map"] == "VIRIDEAN_CITY,":
                    reward += 4
                elif new_state["location"]["map"] == "REDS_HOUSE_1F" or new_state["location"]["map"] == "REDS_HOUSE_2F" or new_state["location"]["map"] == "BLUES_HOUSE":
                    reward -= 1
            else:
                if new_state["location"]["map"] == "OAKS_LAB," or new_state["location"]["map"] == "REDS_HOUSE_1F" or new_state["location"]["map"] == "REDS_HOUSE_2F" or new_state["location"]["map"] == "BLUES_HOUSE":
                    reward -= 1
            #print("Has moved reward")

        elif (new_state["location"] == self.prior_game_stats["location"] and self._read_m(0xD057) == 0):
            self.no_move += 1
            if new_state["location"]["map"] == "ROUTE_1,":
                reward -= 1.5
            else:
                reward -= 1


        # if new_state["location"]["map"] == "PALLET_TOWN," and ("ROUTE_1" not in self.prev_loc):
        #     if (new_state["location"]["y"]) < (self.prior_game_stats["location"]["y"]): #to attempt to go up
        #         reward += 4 #<-Change
        #         #print("Heading entrance of ROUTE_1")
        #     elif (new_state["location"]["y"]) > (self.prior_game_stats["location"]["y"]):
        #         reward -= 2
        #         #print("Away from entrance of ROUTE_1")
        #     elif 0 <= (new_state["location"]["y"]) < 3 and  (9 <new_state["location"]["y"] < 13):
        #         reward += 4.5
            #print("Near entrance of ROUTE_1")

        return reward                  
    
    def _location_reward(self, new_state: dict[str, any]) -> float :
        reward = 0
        if self.in_battle:
            return reward
        #New area 
        if new_state["location"]["map"] not in self.prev_loc:
            self.prev_loc.add(new_state["location"]["map"])
            reward += 20
            if new_state["location"]["map"] == "PALLET_TOWN,":
                reward += 4
            elif new_state["location"]["map"] == "ROUTE_1,":
                reward += 5
            elif new_state["location"]["map"] == "VIRIDEAN_CITY,":
                reward += 6
            elif new_state["location"]["map"] == "REDS_HOUSE_1F" or new_state["location"]["map"] == "REDS_HOUSE_2F" or new_state["location"]["map"] == "BLUES_HOUSE":
                reward -= 20
           #print("Newly visited location")
        # elif new_state["location"]["map"] in self.prev_loc:
        #     reward = 10
            #print("Different location from previously")
        
        # if new_state["location"]["map"] == "OAKS_LAB,":
        #     reward -= 5

        # if new_state["location"]["map"] == "OAKS_LAB," or new_state["location"]["map"] == "REDS_HOUSE_1F" or new_state["location"]["map"] == "REDS_HOUSE_2F" or new_state["location"]["map"] == "BLUES_HOUSE":
        #     reward -= 1

        return reward
    
    def _in_battle_reward(self, new_state) -> float :
        reward = 0
        enemy_hp_df = 0
        if self._read_m(0xD057) != 0 and self.in_battle == False:
            print("Triggered")
            self.in_battle = True
            reward += 5
        else:
            if self._read_m(0xD057) != 0:
                enemy_curr_hp = self._read_hp(0xCFE6) #self._read_m(0xCFE6) | self._read_m(0xCFE7)
                enemy_max_hp = self._read_hp(0xCFF4)  #self._read_m(0xCFF4) | self._read_m(0xCFF5)
                turn_num = self._read_m(0xCCD5) #turn number starts from 0
                battle_hp = self._read_hp(0xD015)
                battle_hp_max = self._read_hp(0xD023)
                battle_left_right = self._read_m(0xCC29) #battle menu cursor on left or right
                battle_button = self._read_m(0xCC26)
                # Move selection: 199, Item selection: 7, Switch Pokemon selection: 3  
                # FIGHT: CurrentMenu = 17, Selected item = 0
                # ITEM: CurrentMenu = 17, Selected Item = 1
                # POKEMON: CurrentMenu = 33, Selected Item = 0
                # RUN    : Current menu = 33, Selected Item = 1
                hp_list = self._read_party_hp()
                party_hp = 0
                for i in hp_list["current"]:
                    party_hp = party_hp + i

                if battle_hp < battle_hp_max:
                    reward -= 1


                #Ensure that its either 33 or 17:
                if battle_left_right in [17, 33] and self.button_pressed == 4:
                    print("In fight")
                    reward += 2
                    if self.no_attack > 0:
                        reward -= 2
                    pokeball_count = self._get_pokeball_count(self._read_items())
                    if pokeball_count == 0:
                        if battle_left_right == 17 and battle_button == 0:
                            print("Fight button has been pressed")
                            reward += 2
                            self.no_attack += 1
                            if self.no_attack > 25:
                                reward -= 2

                    else:
                        #Impossible to reach here unless pokeball is available
                        if battle_left_right == 17 and battle_button == 1:
                            print("Item has been pressed")
                            reward += 3
                            
                elif battle_left_right == 199 and self.button_pressed == 4 :
                    print("entered move selection")
                    reward += 3
                    if self.no_attack > 0:
                        reward -= 3
                    if battle_button == 0:
                        print("Tackle selected")
                        enemy_hp_df = enemy_max_hp - enemy_curr_hp
                        reward += 3
                        if (turn_num != self.prev_turn):
                            print("action made")
                            reward += 5
                        else:
                            reward -= 3
                            self.no_attack += 1
                        
                    elif battle_button == 1:
                        print("Tail whip selected")
                        reward += 2
                        if (turn_num != self.prev_turn):
                            print("action made")
                            reward += 5
                        else:
                            reward -= 2
                            self.no_attack += 1
                    else:
                        print("no button has been pressed")
                        self.no_attack += 1
                    
                    if self._xp_reward(new_state) > 0:
                        reward += 0.5 * self._xp_reward(new_state)
                        self.battle_win +=1

                    if turn_num != self.prev_turn:
                        print("action made reward applied")
                        if self.battle_win > 2:
                            print("Has won 2")
                            reward += 1.5 + 1.5*(enemy_hp_df)
                        else:
                            reward += 30 + 1.5*(enemy_hp_df)
                        self.no_attack = 0
                    
                self.prev_turn = turn_num  
            elif self._read_m(0xD057) == 0 and self.in_battle == True:
                if self.battle_win == 0:
                    print("ran away in shame")
                    reward -= 50
                    self.ran_away = 1
                self.in_battle = False
                self.prev_turn = -1
                self.no_attack = 0


        return reward + self.battle_win*2


    def _get_state(self) -> np.ndarray:
        # Implement your state retrieval logic here
        game_stats = self._generate_game_stats()

        x = np.array([game_stats["location"]["x"]])
        y = np.array([game_stats["location"]["y"]])
        location = np.array([game_stats["location"]["map_id"]])

        hp_list = game_stats["hp"]["current"]
        curr_hp = 0
        for i in hp_list:
            curr_hp = curr_hp + i
        curr_hp = np.array([curr_hp])
        enemy_curr_hp = np.array([self._read_hp(0xCFE6)])
        num_caught = np.array([game_stats["caught_pokemon"]])
        battle_left_right = np.array([self._read_m(0xCC29)]) #battle menu cursor on left or right
        battle_button = np.array([self._read_m(0xCC26)])
        xp = np.array([self._read_party_xp()])
        seen = np.array([self._read_seen_pokemon_count()])

        state = np.concatenate([x, y, 
                                location,
                                curr_hp,
                                enemy_curr_hp,
                                num_caught,
                                battle_button,
                                battle_left_right,
                                seen]).flatten()
   
        return state#, np.array(self.pyboy.game_area())]#state  #[game_stats["location"]["map_id"], game_stats["seen_pokemon"], game_stats["caught_pokemon"]]

    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here
        
        location_reward = self._location_reward(new_state)
        movement_reward = self._movement_reward(new_state)
        seen_reward = self._seen_reward(new_state)
        battle_reward = self._in_battle_reward(new_state)
        xp_reward = self._xp_reward(new_state)
        #print(f"Location: {location_reward}, Movement: {movement_reward}, Seen: {seen_reward}, battle: {battle_reward}, xp: {xp_reward}")

        caught_reward = self._caught_reward(new_state)

        #Try multiplying all by 10?
        total_reward = (movement_reward #* 0.4
                        + location_reward #* 0.8
                        + seen_reward #*0.4
                        + caught_reward #*0.75
                        + battle_reward #* 0.9  
                        + xp_reward * 0 )#*0.5)

        return total_reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here
        if self.steps >= 1000:
            return 1
        elif self.no_move >= 250:
            return 1
        elif self.no_attack >= 150:
            return 1
        elif self._read_hp(0xD015) == 0:
            return 1
        elif self.ran_away == 1:
            return 1
        else:
            return 0
        
        
        # Maybe if we run out of pokeballs...? or a max step count
        #return self.steps >= 1000