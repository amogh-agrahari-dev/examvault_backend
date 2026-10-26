import asyncio
import uuid
import httpx
from database import AsyncSessionLocal
from security import create_access_token
from models.auth_models import User
from sqlalchemy.future import select

BASE_URL = "http://127.0.0.1:8000/api/v1"

async def run_tests():
    print("--- Starting Quick Short Notes API Integration Tests ---")
    
    # Get or create test users
    async with AsyncSessionLocal() as db:
        res_a = await db.execute(select(User).where(User.email == "test_notes_user_a@example.com"))
        user_a = res_a.scalars().first()
        if not user_a:
            user_a = User(email="test_notes_user_a@example.com", full_name="User Alpha", hashed_password="fake")
            db.add(user_a)
            
        res_b = await db.execute(select(User).where(User.email == "test_notes_user_b@example.com"))
        user_b = res_b.scalars().first()
        if not user_b:
            user_b = User(email="test_notes_user_b@example.com", full_name="User Beta", hashed_password="fake")
            db.add(user_b)
            
        await db.commit()
        await db.refresh(user_a)
        await db.refresh(user_b)
        
        user_a_id = str(user_a.id)
        user_b_id = str(user_b.id)
        token_a = create_access_token({"sub": user_a_id, "email": user_a.email})
        token_b = create_access_token({"sub": user_b_id, "email": user_b.email})

    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=15.0) as client:
        # 1. User A creates a note
        note_payload = {
            "title": "Discrete Math Formulas",
            "content": {
                "type": "doc",
                "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": "Euler formula: V - E + F = 2"}]}
                ]
            },
            "content_text": "Euler formula: V - E + F = 2",
            "thumbnail_url": None
        }
        res = await client.post("/notes", json=note_payload, headers=headers_a)
        assert res.status_code == 201, f"Expected 201, got {res.status_code}: {res.text}"
        note_data = res.json()
        note_id = note_data["id"]
        assert note_data["title"] == "Discrete Math Formulas"
        assert note_data["content_text"] == "Euler formula: V - E + F = 2"
        print("✓ Test 1: Note creation succeeded.")

        # 2. User A gets the note
        res = await client.get(f"/notes/{note_id}", headers=headers_a)
        assert res.status_code == 200
        assert res.json()["title"] == "Discrete Math Formulas"
        print("✓ Test 2: Note retrieval by owner succeeded.")

        # 3. User B tries to get User A's note (Authorization check)
        res = await client.get(f"/notes/{note_id}", headers=headers_b)
        assert res.status_code == 404, f"Expected 404 for unauthorized read, got {res.status_code}"
        print("✓ Test 3: Unauthorized note access rejected with 404.")

        # 4. User A updates note (Autosave simulation)
        update_payload = {
            "title": "Discrete Math Graph Theory",
            "content": {
                "type": "doc",
                "content": [
                    {"type": "heading", "attrs": {"level": 2}, "content": [{"type": "text", "text": "Planar Graphs"}]},
                    {"type": "paragraph", "content": [{"type": "text", "text": "Sum of degrees = 2 * |E|"}]}
                ]
            },
            "content_text": "Planar Graphs: Sum of degrees = 2 * |E|"
        }
        res = await client.put(f"/notes/{note_id}", json=update_payload, headers=headers_a)
        assert res.status_code == 200
        assert res.json()["title"] == "Discrete Math Graph Theory"
        print("✓ Test 4: Note autosave/update succeeded.")

        # 5. User B tries to update User A's note
        res = await client.put(f"/notes/{note_id}", json={"title": "Hacked Title"}, headers=headers_b)
        assert res.status_code == 404
        print("✓ Test 5: Unauthorized update rejected.")

        # 6. User A lists notes with search
        res = await client.get("/notes?search=Euler", headers=headers_a)
        assert res.status_code == 200
        # Search for updated content
        res = await client.get("/notes?search=Planar", headers=headers_a)
        assert res.status_code == 200
        items = res.json()
        assert len(items) >= 1
        assert items[0]["id"] == note_id
        print("✓ Test 6: Note listing and search succeeded.")

        # 7. User A uploads an image
        fake_png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        files = {"file": ("graph.png", fake_png_data, "image/png")}
        res = await client.post(f"/notes/upload-image?note_id={note_id}", files=files, headers=headers_a)
        assert res.status_code == 200, f"Image upload failed: {res.text}"
        img_data = res.json()
        image_id = img_data["id"]
        assert img_data["url"] == f"/api/v1/notes/images/{image_id}"
        print("✓ Test 7: Image upload succeeded with image ID:", image_id)

        # 8. User A fetches the image using ?token=
        res = await client.get(f"/notes/images/{image_id}?token={token_a}")
        assert res.status_code == 200, f"Image fetch failed: {res.status_code}"
        assert len(res.content) > 0
        print("✓ Test 8: Image fetch with owner token succeeded.")

        # 9. User B tries to fetch User A's image
        res = await client.get(f"/notes/images/{image_id}?token={token_b}")
        assert res.status_code == 403, f"Expected 403 for unauthorized image access, got {res.status_code}"
        print("✓ Test 9: Unauthorized image access prevented with 403.")

        # 10. User B tries to delete User A's note
        res = await client.delete(f"/notes/{note_id}", headers=headers_b)
        assert res.status_code == 404
        print("✓ Test 10: Unauthorized note deletion rejected.")

        # 11. User A deletes note
        res = await client.delete(f"/notes/{note_id}", headers=headers_a)
        assert res.status_code == 204
        print("✓ Test 11: Note deletion by owner succeeded.")

        # 12. Verify note is gone
        res = await client.get(f"/notes/{note_id}", headers=headers_a)
        assert res.status_code == 404
        print("✓ Test 12: Note verified gone (404).")

    print("\nALL 12 BACKEND INTEGRATION TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    asyncio.run(run_tests())
