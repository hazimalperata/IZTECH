package src.quiz;



import utils.Rand;

import java.util.ArrayList;
import java.util.List;

import src.question.IQuestion;
import src.question.Listening;
import src.question.Reading;
import src.question.Speaking;
import src.question.WordMatching;

public class Quiz {
    private final int MAX_QUESTION_NUMBER = 15;
    private final int MIN_QUESTION_NUMBER = 8;

    private int numberOfReadingQuestion = 0;
    private int numberOfListeningQuestion = 0;
    private int numberOfSpeakingQuestion = 0;
    private int numberOfWordMatchingQuestion = 0;
    private String name;
    private List<IQuestion> questions;


    public Quiz(String name) {
        this.name = name;
        createQuestions();
    }

    public Quiz(String name, List<IQuestion> questions) {
        this.name = name;
        this.questions = questions;
        setNumberOfQuestionWithType();
    }

    private void createQuestions() {
        Rand rand = new Rand();
        this.questions = new ArrayList<>();
        int count = rand.getInt(MIN_QUESTION_NUMBER, MAX_QUESTION_NUMBER);
        for (int i = 0; i < count; i++) {
            int typeCount = rand.getInt(4);
            switch (typeCount) {
                case 0 -> {
                    Reading reading = new Reading();
                    questions.add(reading);
                    numberOfReadingQuestion++;
                }
                case 1 -> {
                    Listening listening = new Listening();
                    questions.add(listening);
                    numberOfListeningQuestion++;
                }
                case 2 -> {
                    Speaking speaking = new Speaking();
                    questions.add(speaking);
                    numberOfSpeakingQuestion++;
                }
                case 3 -> {
                    WordMatching wordMatching = new WordMatching();
                    questions.add(wordMatching);
                    numberOfWordMatchingQuestion++;
                }
            }
        }
    }

    public String getName() {
        return name;
    }

    private void setNumberOfQuestionWithType() {
        for (IQuestion question : questions) {
            if (question instanceof Reading) {
                numberOfReadingQuestion++;
            } else if (question instanceof Speaking) {
                numberOfSpeakingQuestion++;
            } else if (question instanceof Listening) {
                numberOfListeningQuestion++;
            } else {
                numberOfWordMatchingQuestion++;
            }
        }
    }

    public int solve() {
        Rand rand = new Rand();
        int totalPoint = 0;
        for (IQuestion question : questions) {
            int isAnswered = rand.getInt(1);
            if (isAnswered == 1) {
                totalPoint += question.getPoints();
            }
        }
        return totalPoint;
    }

    public List<IQuestion> getQuestions() {
        return new ArrayList<IQuestion>(questions);
    }

    public IQuestion getQuestion(int i) {
        return questions.get(i);
    }

    public int getNumberOfQuestions() {
        return this.questions.size();
    }

    public int getNumberOfReadingQuestion() {
        return numberOfReadingQuestion;
    }

    public int getNumberOfListeningQuestion() {
        return numberOfListeningQuestion;
    }

    public int getNumberOfSpeakingQuestion() {
        return numberOfSpeakingQuestion;
    }

    public int getNumberOfWordMatchingQuestion() {
        return numberOfWordMatchingQuestion;
    }

    @Override
    public String toString() {
        String message = name + ",";
        String readingQuestion = numberOfReadingQuestion == 0 ? "" : numberOfReadingQuestion + "R:";
        String listeningQuestion = numberOfListeningQuestion == 0 ? "" : numberOfListeningQuestion + "L:";
        String speakingQuestion = numberOfSpeakingQuestion == 0 ? "" : numberOfSpeakingQuestion + "S:";
        String wordMatchingQuestion = numberOfWordMatchingQuestion == 0 ? "" : numberOfWordMatchingQuestion + "W";
        message = message + readingQuestion + listeningQuestion + speakingQuestion + wordMatchingQuestion;
        if (message.endsWith(":")){
            message = message.substring(0, message.length()-1);
        }
        return message;
    }

}
