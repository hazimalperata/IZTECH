package src.fileIO;

import src.language.Language;
import src.league.ILeague;
import src.league.League;
import src.question.IQuestion;
import src.question.Listening;
import src.question.Reading;
import src.question.Speaking;
import src.question.WordMatching;
import src.quiz.Quiz;
import src.unit.Unit;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class LanguagesFileManagement {
    private LanguagesFileIO languagesFileIO = new LanguagesFileIO();
    private List<Language> languagesList = new ArrayList<Language>();

    public LanguagesFileManagement() {
    }

    public boolean isExistFile() {
        return languagesFileIO.isExistFile();
    }

    public List<Language> getLanguagesListFromFile(List<List<ILeague>> leagues) {
        List<Unit> units = new ArrayList<Unit>();
        List<Integer> unitsIndex = new ArrayList<Integer>();
        List<Integer> quizzesIndex = new ArrayList<>();
        List<Quiz> quizzes = new ArrayList<Quiz>();
        List<IQuestion> questions = new ArrayList<IQuestion>();
        List<String> lineList = languagesFileIO.readFile();
        
        int index = 0;
        for (String line : lineList) {
            String[] values = line.split(",");
            String languageName = values[0];
            for (int i = 1; i < values.length; i++) {
                if (values[i].startsWith("Unit")) {
                    unitsIndex.add(i);
                }
            }
            for (int i = 0; i < unitsIndex.size(); i++) {
                String[] quizArray;
                if (i == unitsIndex.size() - 1) {
                    quizArray = Arrays.copyOfRange(values, unitsIndex.get(i), values.length);
                } else {
                    quizArray = Arrays.copyOfRange(values, unitsIndex.get(i), unitsIndex.get(i + 1));
                }
                for (int quizIndex = 0; quizIndex < quizArray.length; quizIndex++) {
                    if (quizArray[quizIndex].startsWith("Quiz")) {
                        quizzesIndex.add(quizIndex);
                    }
                }
                for (int j = 0; j < quizzesIndex.size(); j++) {
                    String[] questionArray;
                    if (j == quizzesIndex.size() - 1) {
                        questionArray = Arrays.copyOfRange(quizArray, quizzesIndex.get(j), quizArray.length);
                    } else {
                        questionArray = Arrays.copyOfRange(quizArray, quizzesIndex.get(j), quizzesIndex.get(j + 1));
                    }
                    String[] questionType = questionArray[1].split(":");
                    for (int k = 0; k < questionType.length; k++) {
                        if (questionType[k].endsWith("R")) {
                            for (int l = 0; l < Integer.parseInt(String.valueOf(questionType[k].charAt(0))); l++) {
                                questions.add(new Reading());
                            }
                        } else if (questionType[k].endsWith("L")) {
                            for (int l = 0; l < Integer.parseInt(String.valueOf(questionType[k].charAt(0))); l++) {
                                questions.add(new Listening());
                            }
                        } else if (questionType[k].endsWith("W")) {
                            for (int l = 0; l < Integer.parseInt(String.valueOf(questionType[k].charAt(0))); l++) {
                                questions.add(new WordMatching());
                            }
                        } else {
                            for (int l = 0; l < Integer.parseInt(String.valueOf(questionType[k].charAt(0))); l++) {
                                questions.add(new Speaking());
                            }
                        }
                    }
                    Quiz quiz = new Quiz("Quiz %s".formatted(j + 1), new ArrayList<IQuestion>(questions));
                    questions.clear();
                    quizzes.add(quiz);

                }
                Unit unit = new Unit("Unit %s".formatted(i + 1), new ArrayList<Quiz>(quizzes));
                quizzes.clear();
                units.add(unit);
                quizzesIndex.clear();
            }

            Language language = new Language(languageName, new ArrayList<Unit>(units), leagues.get(index));
            units.clear();
            languagesList.add(language);
            unitsIndex.clear();
            index +=1 ;
        }
        return languagesList;

    }

    public void writeLanguages(List<Language> languages){
        List<String> languagesRow = new ArrayList<>();
        for (Language language : languages) {
            languagesRow.add(language.toString());
        }
        languagesFileIO.writeFile(languagesRow);
    }
    /*
    public void writeLanguages(List<Language> languages) {
        List<String> languagesRow = new ArrayList<>();
        for (Language language : languages) {
            String row = language.getName() + ",";
            for (int unitNum = 0; unitNum < language.getNumberOfUnits(); unitNum++) {
                Unit unit = language.getUnit(unitNum);
                row = row + "Unit " + (unitNum + 1) + ",";
                for (int quizNum = 0; quizNum < unit.getNumberOfQuizzes(); quizNum++) {
                    Quiz quiz = unit.getQuiz(quizNum);
                    row = row + "Quiz " + (quizNum + 1) + "," +
                            (quiz.getNumberOfReadingQuestion() == 0 ? "" : quiz.getNumberOfReadingQuestion() + "R:") +
                            (quiz.getNumberOfListeningQuestion() == 0 ? "" : quiz.getNumberOfListeningQuestion() + "L:") +
                            (quiz.getNumberOfSpeakingQuestion() == 0 ? "" : quiz.getNumberOfSpeakingQuestion() + "S:") +
                            (quiz.getNumberOfWordMatchingQuestion() == 0 ? "" : quiz.getNumberOfWordMatchingQuestion() + "W");
                    if (row.endsWith(":")) {
                        row = row.substring(0, row.length() - 1);
                    }
                    row += ",";
                }
            }
            languagesRow.add(row.substring(0, row.length() - 1));
        }
        languagesFileIO.writeFile(languagesRow);
    }
*/
}
