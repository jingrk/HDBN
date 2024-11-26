File path and file description:

》Root path

**song_list_7digital.csv**:  This file contains the music information provided by the LJ2M [1] dataset, including artist, music title, and music ID.

**emotion_map.json**:  This file contains the emotion words corresponding to the user emotion indexes.

|——\\**EmoMusicLJ**\\

|————**user_music_interactions.csv**: The content of this file consists of all user-music interaction records, provided in the form of (user, music, user-emotion) triples. (**Note**: The values represent the indexes of the user, music, and user emotion tags, not the original IDs.)
	|————**index2userID.json**: This file contains the mapping from user indexes to original IDs, provided in dictionary format.

|————**index2itemID.json**： This file contains the mapping from music indexes to original IDs, provided in dictionary format.

|————**song_meta_info.json**:  This file contains the metadata corresponding to the original music IDs, provided in the form of {music_ID: [artist, music title, genre, release year], ...}.

|————**songs_audio_emo.npy**: This file contains the mood vectors for the music, with each row corresponding to the music index and representing a 9-dimensional vector. The 9 dimensions represent the proportions (distributions) of the moods: Amazement, Solemnity, Tenderness, Nostalgia, Calmness, Power, Joyful activation, Tension, and Sadness.     Music moods were labeled using a model trained on the Emotify dataset [2]. Alternatively, other methods can be used to label the music moods based on the audio features.
