import os, sys
abs_path = os.path.dirname(os.path.dirname(os.path.abspath('.')))
print(abs_path)
sys.path.append(abs_path)

from weat import read_token, plot_heatmap, SynopsisWEAT
from gensim.models import Word2Vec

dir_path = f'{abs_path}\\data\\synopsis'

def get_tokens(file_names:list):
    tokens = []
    for file in file_names:
        file_path = dir_path + "\\" + file
        print(file_path)
        tokens.append(read_token(file_path))
        tokens.append(file_path)
    return tokens

def get_w2v():
    tokenized = get_tokens(['synopsis.txt'])

    # tokenized에 담긴 데이터를 가지고 나만의 Word2Vec을 생성합니다. (Gensim 4.0 기준)
    model = Word2Vec(tokenized, vector_size=100, window=5, min_count=3, sg=0)
    return model

model = get_w2v()

targets_filenames = ['synopsis_art.txt', 'synopsis_gen.txt']
targets = get_tokens(targets_filenames)

attribute_filenames = [
    'synopsis_SF.txt', 'synopsis_family.txt', 'synopsis_show.txt',
    'synopsis_horror.txt', 'synopsis_etc.txt', 'synopsis_documentary.txt',
    'synopsis_drama.txt', 'synopsis_romance.txt', 'synopsis_musical.txt',
    'synopsis_mystery.txt', 'synopsis_crime.txt', 'synopsis_historical.txt',
    'synopsis_western.txt', 'synopsis_adult.txt', 'synopsis_thriller.txt',
    'synopsis_animation.txt', 'synopsis_action.txt', 'synopsis_adventure.txt',
    'synopsis_war.txt', 'synopsis_comedy.txt', 'synopsis_fantasy.txt']
attributes = get_tokens(attribute_filenames)

genre_name = [
    'SF', '가족', '공연', '공포(호러)', '기타', '다큐멘터리', '드라마', '멜로로맨스',
    '뮤지컬', '미스터리', '범죄', '사극', '서부극(웨스턴)', '성인물(에로)', '스릴러',
    '애니메이션', '액션', '어드벤처', '전쟁', '코미디', '판타지']

syweat = SynopsisWEAT()
syweat.set_model(model)
syweat.make_target(targets)
syweat.make_excluded_words()
syweat.make_attributes(attributes, genre_name)
syweat.make_matrix()

ax = plot_heatmap(syweat.matrix, genre_name, "heatmap.png")

