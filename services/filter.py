from geopy.distance import geodesic
import logging
logger = logging.getLogger(__name__)

def is_within_radius(user_lat, user_lon, target_lat, target_lon, radius_km=10):
    try:
        return geodesic((user_lat, user_lon), (target_lat, target_lon)).km <= radius_km
    except Exception as e:
        logger.warning(f"❗ 거리 계산 실패: {e}")
        return False

def apply_filter(state: dict):
    try:
        lat = state.get("latitude")
        lon = state.get("longitude")
        restrictions = state.get("restrictions", [])
        candidates = state.get("search_results", [])

        def is_within_radius(user_lat, user_lon, target_lat, target_lon, radius_km=3):
            try:
                return geodesic((user_lat, user_lon), (target_lat, target_lon)).km <= radius_km
            except Exception as e:
                logger.warning("❗ 거리 계산 실패: %s", e)
                return False

        filtered = []
        for item in candidates:
            if not item: continue
            loc = item.get("location")
            if loc and not is_within_radius(lat, lon, loc.get("lat"), loc.get("lon")):
                continue

            name = item.get("menu", "").lower()
            if any(res.lower() in name for res in restrictions):
                continue

            filtered.append(item)
            
        # ✅ 상위 3개만 자르기 (우선순위가 정렬되어 있다고 가정)
        filtered_top3 = filtered[:3]
        logger.info("✅ 필터링 결과 수: %d", len(filtered))
        return {**state, "filtered_results": filtered}
    except Exception as e:
        logger.exception("❌ [apply_filter] 오류 발생: %s", e)
        raise