from backend.database import execute_query, fetch_query

class Attendance:
    """Handles attendance-related database operations."""

    @staticmethod
    def mark_attendance(person_name, status):
        query = """
        INSERT INTO attendance (person_name, date, timestamp, status)
        VALUES (%s, CURDATE(), NOW(), %s)
        """
        execute_query(query, (person_name, status))

    @staticmethod
    def get_attendance_records():
        query = "SELECT id, person_name, date, timestamp, status FROM attendance ORDER BY date DESC, timestamp DESC"
        return fetch_query(query)

class FaceEnrollment:
    """Handles face enrollment-related database operations."""

    @staticmethod
    def enroll_face(person_name, image_path):
        query = """
        INSERT INTO face_data (person_name, image_path, created_at)
        VALUES (%s, %s, NOW())
        """
        execute_query(query, (person_name, image_path))

    @staticmethod
    def get_all_enrolled_faces():
        query = "SELECT person_name, image_path FROM face_data"
        return fetch_query(query)

# Example usage:
if __name__ == "__main__":
    print("Attendance Records:", Attendance.get_attendance_records())
    print("Enrolled Faces:", FaceEnrollment.get_all_enrolled_faces())
