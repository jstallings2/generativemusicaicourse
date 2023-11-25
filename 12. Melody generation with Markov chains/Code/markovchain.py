import os
import numpy as np
from music21 import metadata, note, stream, converter

KERN_DATASET_PATH = "./deutschl/test/"


class MarkovChainMelodyGenerator:
    """
    Represents a Markov Chain model for melody generation.
    """

    def __init__(self, states):
        """
        Initialize the MarkovChain with a list of states.

        Parameters:
            states (list of tuples): A list of possible (pitch, duration)
                pairs.
        """
        self.states = states
        self.initial_probabilities = np.zeros(len(states))
        self.transition_matrix = np.zeros((len(states), len(states)))
        self._state_indexes = {state: i for (i, state) in enumerate(states)}

    def train(self, notes):
        """
        Train the model based on a list of notes.

        Parameters:
            notes (list): A list of music21.note.Note objects.
        """
        self._calculate_initial_probabilities(notes)
        self._calculate_transition_matrix(notes)
        print('final init probabilities: ', self.initial_probabilities)
        print('final transition matrix: ', self.transition_matrix)

    def generate(self, length):
        """
        Generate a melody of a given length.

        Parameters:
            length (int): The length of the sequence to generate.

        Returns:
            melody (list of tuples): A list of generated states.
        """
        melody = [self._generate_starting_state()]
        for _ in range(1, length):
            melody.append(self._generate_next_state(melody[-1]))
        return melody

    def _calculate_initial_probabilities(self, notes):
        """
        Calculate the initial probabilities from the provided notes.

        Parameters:
            notes (list): A list of music21.note.Note objects.
        """
        for note in notes:
            self._increment_initial_probability_count(note)
        self._normalize_initial_probabilities()

    def _increment_initial_probability_count(self, note):
        """
        Increment the probability count for a given note.

        Parameters:
            note (music21.note.Note): A note object.
        """
        state = (note.pitch.nameWithOctave, note.duration.quarterLength)
        self.initial_probabilities[self._state_indexes[state]] += 1

    def _normalize_initial_probabilities(self):
        """
        Normalize the initial probabilities array such that the sum of all
        probabilities equals 1.
        """
        total = np.sum(self.initial_probabilities)
        if total:
            self.initial_probabilities /= total
        self.initial_probabilities = np.nan_to_num(self.initial_probabilities)

    def _calculate_transition_matrix(self, notes):
        """
        Calculate the transition matrix from the provided notes.

        Parameters:
            notes (list): A list of music21.note.Note objects.
        """
        for i in range(len(notes) - 1):
            self._increment_transition_count(notes[i], notes[i + 1])
        self._normalize_transition_matrix()

    def _increment_transition_count(self, current_note, next_note):
        """
        Increment the transition count from current_note to next_note.

        Parameters:
            current_note (music21.note.Note): The current note object.
            next_note (music21.note.Note): The next note object.
        """
        state = (
            current_note.pitch.nameWithOctave,
            current_note.duration.quarterLength,
        )
        next_state = (
            next_note.pitch.nameWithOctave,
            next_note.duration.quarterLength,
        )
        self.transition_matrix[
            self._state_indexes[state], self._state_indexes[next_state]
        ] += 1
        print(' updated transition: ', self.transition_matrix)

    def _normalize_transition_matrix(self):
        """
        This method normalizes each row of the transition matrix so that the
        sum of probabilities in each row equals 1. This is essential for the rows
        of the matrix to represent probability distributions of
        transitioning from one state to the next.
        """

        # Calculate the sum of each row in the transition matrix.
        # These sums represent the total count of transitions from each state
        # to any other state.
        row_sums = self.transition_matrix.sum(axis=1)

        # Use np.errstate to ignore any warnings that arise during division.
        # This is necessary because we might encounter rows with a sum of 0,
        # which would lead to division by zero.
        with np.errstate(divide="ignore", invalid="ignore"):
            # Normalize each row by its sum. np.where is used here to handle
            # rows where the sum is zero.
            # If the sum is zero (no transitions from that state), np.where
            # ensures that the row remains a row of zeros instead of turning
            # into NaNs due to division by zero.
            self.transition_matrix = np.where(
                row_sums[:, None],  # Condition: Check each row's sum.
                # True case: Normalize if sum is not zero.
                self.transition_matrix / row_sums[:, None],
                0,  # False case: Keep as zero if sum is zero.
            )

    def _generate_starting_state(self):
        """
        Generate a starting state based on the initial probabilities.

        Returns:
            A state from the list of states.
        """
        initial_index = np.random.choice(
            list(self._state_indexes.values()), p=self.initial_probabilities
        )
        return self.states[initial_index]

    def _generate_next_state(self, current_state):
        """
        Generate the next state based on the transition matrix and the current
        state.

        Parameters:
            current_state: The current state in the Markov Chain.

        Returns:
            The next state in the Markov Chain.
        """
        if self._does_state_have_subsequent(current_state):
            index = np.random.choice(
                list(self._state_indexes.values()),
                p=self.transition_matrix[self._state_indexes[current_state]],
            )
            return self.states[index]
        return self._generate_starting_state()

    def _does_state_have_subsequent(self, state):
        """
        Check if a given state has a subsequent state in the transition matrix.

        Parameters:
            state: The state to check.

        Returns:
            True if the state has a subsequent state, False otherwise.
        """
        return self.transition_matrix[self._state_indexes[state]].sum() > 0


def create_training_data(dataset_path):
    """
    Creates a list of sample training notes for the melody of "Twinkle
    Twinkle Little Star."

    Returns:
        - list: A list of music21.note.Note objects.
    """
    if not dataset_path:
        data = [
            note.Note("C5", quarterLength=1),
            note.Note("C5", quarterLength=1),
            note.Note("G5", quarterLength=1),
            note.Note("G5", quarterLength=1),
            note.Note("A5", quarterLength=1),
            note.Note("A5", quarterLength=1),
            note.Note("G5", quarterLength=2),
            note.Note("F5", quarterLength=1),
            note.Note("F5", quarterLength=1),
            note.Note("E5", quarterLength=1),
            note.Note("E5", quarterLength=1),
            note.Note("D5", quarterLength=1),
            note.Note("D5", quarterLength=1),
            note.Note("C5", quarterLength=2),
        ]
        states = [
        ("C5", 1),
        ("D5", 1),
        ("E5", 1),
        ("F5", 1),
        ("G5", 1),
        ("A5", 1),
        ("C5", 2),
        ("D5", 2),
        ("E5", 2),
        ("F5", 2),
        ("G5", 2),
        ("A5", 2),
        ]
        return data, states 
    # read in the notes from the song
    song = load_song_in_kern(dataset_path)
    data = []
    states = set()

    # for each note,
    for el in song.recurse().notes:
        print(el.pitch.nameWithOctave,
            el.duration.quarterLength)
        # add the note/dur tuple to the "states" if it's a note we haven't seen
        states.add((el.pitch.nameWithOctave, el.duration.quarterLength))
        # add the m21.note.Note to the "data"
        data.append(el)

    # print ('data: ', data)
    # print('states: ', states)
    return data, list(states)

def load_song_in_kern(dataset_path):
    # go through all the files in dataset and load them with music21
    # with the return in here we just do the first song
    # @returns a Music21 Stream
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == 'krn':
                return converter.parse(os.path.join(path, file))
            
    print('Error loading score')

def preprocess(dataset_path):
    pass

    # load the folk songs
    print("Loading song...")
    song = load_song_in_kern(dataset_path)

    # filter out songs that have non-acceptable durations
    # (DONT NEED)

    # transpose songs to Cmaj/Amin
    # (DONT NEED?)

    # encode songs with music time series representation

    # save songs to text file


def visualize_melody(melody):
    """
    Visualize a sequence of (pitch, duration) pairs using music21.

    Parameters:
        - melody (list): A list of (pitch, duration) pairs.
    """
    print(melody)
    score = stream.Score()
    score.metadata = metadata.Metadata(title="Markov Chain Melody")
    part = stream.Part()
    for n, d in melody:
        part.append(note.Note(n, quarterLength=d))
    score.append(part)
    score.show()


def main():
    """Main function for training the chain, generating a melody, and
    visualizing the result."""

    training_data, states = create_training_data(KERN_DATASET_PATH)

    
    model = MarkovChainMelodyGenerator(states)
    model.train(training_data)

    generated_melody = model.generate(40)
    visualize_melody(generated_melody)


if __name__ == "__main__":
    main()
