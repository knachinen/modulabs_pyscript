## Module

- SynopsisWEAT - class
- cos_sim()
- weat_score()
- read_token()
- plot_heatmap()

### Process

```python
syweat = SynopsisWEAT()
syweat.set_model(model)
syweat.make_target([art, gen])
syweat.make_excluded_words()
syweat.make_attributes(genre_all, genre_name)
syweat.make_matrix()
```


Heatmap
```python
ax = plot_heatmap(syweat.matrix, genre_name)
```

![](heatmap.png)