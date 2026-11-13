# Release history / История версий

## English

The public repository is [biscet/DFlash-Coach](https://github.com/biscet/DFlash-Coach).
Version 1.0.0 starts this source repository's history independently of model
weights and dataset versions.

- `main` contains current development.
- The branch `v1.0.0` preserves the original first release.
- Release snapshots use branches without tags.
- Future releases get a new version branch; existing snapshots stay unchanged.

Before the next release, update VERSION and CHANGELOG and review the intended
source changes. Run the relevant tests and the source privacy check. Commit the
reviewed source before creating the next snapshot. For example, for 1.1.0:

```bash
git switch main
python3 -m unittest discover -s tests -v
python3 coach.py privacy-check
git status --short
git diff --cached
git commit -m "Release DFlash Coach v1.1.0"
git branch v1.1.0
git push origin main refs/heads/v1.1.0
```

Stage the intended files before committing. Keep private datasets, weights,
credentials and run artifacts out of Git. Do not force-push over a published
release snapshot.

## Русский

Публичный репозиторий — [biscet/DFlash-Coach](https://github.com/biscet/DFlash-Coach).
Версия 1.0.0 начинает историю исходников независимо от версий весов и датасетов.

- `main` содержит текущую разработку.
- Ветка `v1.0.0` сохраняет исходный первый релиз.
- Срезы релизов хранятся в ветках без тегов.
- Для следующего выпуска создаётся новая ветка; существующие срезы не меняются.

Перед выпуском обновите VERSION и CHANGELOG, проверьте состав изменений,
необходимые тесты и privacy-check. Сначала сохраните исходники в commit, затем
создайте ветку следующей версии. Команды выше показывают пример для 1.1.0.
Нужные файлы следует заранее добавить в staging.

Не добавляйте в Git приватные датасеты, веса, ключи и результаты рабочих run.
Не перезаписывайте опубликованные срезы через force-push.
