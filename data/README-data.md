
File path and file description:

》Root path

**song_list_7digital.csv**:  This file contains the music information provided by the LJ2M [1] dataset, including artist, music title, and music ID.

**emotion_map.json**:  This file contains the emotion words corresponding to the user emotion indexes.

|——\\**EmoMusicLJ**

|————\\**user_music_interactions.csv**: The content of this file consists of all user-music interaction records, provided in the form of (user, music, user-emotion) triples. (**Note**: The values represent the indexes of the user, music, and user emotion tags, not the original IDs.)

|————\\**index2userID.json**: This file contains the mapping from user indexes to original IDs, provided in dictionary format.

|————\\**index2itemID.json**： This file contains the mapping from music indexes to original IDs, provided in dictionary format.

|————\\**song_meta_info.json**:  This file contains the metadata corresponding to the original music IDs, provided in the form of {music_ID: [artist, music title, genre, release year], ...}.

|————\\**songs_audio_emo.npy**: This file contains the mood vectors for the music, with each row corresponding to the music index and representing a 9-dimensional vector. The 9 dimensions represent the proportions (distributions) of the moods: Amazement, Solemnity, Tenderness, Nostalgia, Calmness, Power, Joyful activation, Tension, and Sadness.     Music moods were labeled using a model trained on the Emotify dataset [2]. Alternatively, other methods can be used to label the music moods based on the audio features.

|——\\**EmoMusicLJ-small**
EmoMusicLJ-small is derived from EmoMusicLJ. we performed an additional mapping of user and music indexes between EmoMusicLJ-small and EmoMusicLJ. To get the original user and music IDs of EmoMusicLJ-small, you can first query the corresponding indexes in EmoMusicLJ, then look up the corresponding original user and music IDs of EmoMusicLJ.

|————\\**user_music_interactions.csv**: The content of this file consists of all user-music interaction records, provided in the form of (user, music, user-emotion) triples. (**Note**: The values represent the indexes of the user, music, and user emotion tags, not the original IDs.)

|————\\**song_meta_info.json**: This file is the same as the one in the EmoMusicLJ folder.

|————\\**user_index_small2large.json**： This file contains the mapping of user indexes from EmoMusicLJ-small to EmoMusicLJ, provided in dict format.

|————\\**song_index_small2large.json**: This file contains the mapping of  music indexes from EmoMusicLJ-small to EmoMusicLJ, provided in dict format.

|————\\**songs_audio_emo.npy**: The music moods matrix. Each row corresponding to one music track.


[1] Liu, Jen-Yu, Sung-Yen Liu, and Yi-Hsuan Yang. LJ2M dataset: Toward better understanding of music listening behavior and user mood. *2014 IEEE International Conference on Multimedia and Expo (ICME)*. IEEE, 2014.

[2] Aljanaki, Anna, Frans Wiering, and Remco C. Veltkamp. Studying emotion induced by music through a crowdsourcing game. *Information Processing & Management* 52.1 (2016): 115-128.
