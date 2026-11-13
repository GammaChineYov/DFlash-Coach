# Release history / История версий

## English

Version 1.0.0 starts the history of this code repository. It is separate from
the version numbers of any model weights or datasets.

- `main` contains the current source.
- The branch `v1.0.0` preserves the first release.
- The annotated tag `v1.0.0` records the same release commit.
- New releases get a new branch and tag. Existing release branches stay unchanged.

Before the next release, switch to main, update VERSION and CHANGELOG, and run
the tests and privacy check. Review the staged files. Commit the reviewed source
before creating the next snapshot. For example, after preparing version 1.1.0:

```bash
git switch main
python3 -m unittest discover -s tests -v
python3 coach.py privacy-check
git status --short
git diff --cached
git commit -m "Release DFlash Coach v1.1.0"
git branch v1.1.0
git tag -a v1.1.0 -m "DFlash Coach v1.1.0"
```

Stage the intended changes before the commit. Do not use force-push to replace
an existing public release. Do not copy private run files into a version branch.
Use explicit ref names when a command could confuse the branch and tag.

No remote is required for local history. When publication is approved, add the
chosen GitHub repository and push main, the release branch and the release tag.
For the first release, after adding that remote:

```bash
git push -u origin main
git push origin refs/heads/v1.0.0 refs/tags/v1.0.0
```

The supplied local repository uses a neutral project author. Set your preferred
public author identity yourself before making future commits.

## Русский

Версия 1.0.0 начинает историю этого репозитория кода. Она не продолжает
нумерацию весов или датасетов.

- main содержит актуальные исходники.
- Ветка v1.0.0 сохраняет первый релиз.
- Аннотированный тег v1.0.0 указывает на тот же commit.
- Для каждого следующего выпуска создаются новая ветка и тег. Старый срез не меняется.

Перед новым выпуском перейдите на main, измените VERSION и CHANGELOG,
запустите тесты и privacy-check. Проверьте staged-файлы. Сначала сохраните
проверенные исходники в commit, затем создайте ветку и тег следующей версии.
Команды выше показывают пример для 1.1.0. Нужные изменения следует заранее
добавить в staging.

Не перезаписывайте опубликованные релизы через force-push. Рабочие данные
обучения не должны попадать в ветку версии. Если имя ветки совпадает с тегом,
в неоднозначной команде используйте полный ref.

Для локальной истории remote не нужен. После отдельного решения о публикации
добавьте выбранный GitHub-репозиторий и отправьте main, ветку и тег релиза.
Наличие локальных веток не означает, что проект уже опубликован.

В подготовленном репозитории указан нейтральный автор проекта. Перед будущими
commit настройте ту публичную подпись автора, которую хотите использовать.
