package src.unit;


import utils.Rand;

import java.util.ArrayList;
import java.util.List;

import src.quiz.Quiz;

public class Unit {
    private final int MAX_QUIZZES_NUMBER = 10;
    private final int MIN_QUIZZES_NUMBER = 1;

    private String name;
    private List<Quiz> quizzes;


    public Unit(String name) {
        this.name = name;
        createQuizzes();
    }

    public Unit(String name, List<Quiz> quizzes) {
        this.quizzes = quizzes;
        this.name = name;
    }

    private void createQuizzes() {
        Rand rand = new Rand();
        this.quizzes = new ArrayList<>();
        int count = rand.getInt(MIN_QUIZZES_NUMBER, MAX_QUIZZES_NUMBER);
        for (int i = 0; i < count; i++) {
            Quiz quiz = new Quiz("Quiz %s".formatted(i + 1));
            quizzes.add(quiz);
        }

    }

    public List<Quiz> getQuizzes() {
        return new ArrayList<Quiz>(quizzes);
    }

    public Quiz getQuiz(int i) {
        return quizzes.get(i);
    }

    public int getNumberOfQuizzes() {
        return this.quizzes.size();
    }

    public String getName() {
        return name;
    }


    public int getUnitNumber() {
        return Integer.parseInt(this.name.split(" ")[1]);
    }

    @Override
    public String toString() {
        String message = name + ",";
        for (Quiz quiz : quizzes) {
            message = message + quiz + ",";
        }
        return message.substring(0, message.length() - 1);
    }
}
