DELIMITER $$

CREATE PROCEDURE AddBonus(
    IN in_user_id INT,
    IN in_project_name VARCHAR(255),
    IN in_score INT
)
BEGIN
    DECLARE proj_id INT;

    -- Check if the project already exists
    SELECT id INTO proj_id
    FROM projects
    WHERE name = in_project_name
    LIMIT 1;

    -- If project doesn't exist, insert it
    IF proj_id IS NULL THEN
        INSERT INTO projects (name) VALUES (in_project_name);
        SET proj_id = LAST_INSERT_ID();
    END IF;

    -- Insert the correction
    INSERT INTO corrections (user_id, project_id, score)
    VALUES (in_user_id, proj_id, in_score);
END$$

DELIMITER ;
